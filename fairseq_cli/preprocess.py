#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

from collections import Counter
from itertools import zip_longest
import logging
from multiprocessing import Pool
import os
import shutil
import sys

from fairseq import options, tasks, utils
from fairseq.data import indexed_dataset
from fairseq.binarizer import Binarizer


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.preprocess')


def convert_seqtag_format_to_src_tgt(files_dir, split):
    """
    Converts earch sequence tagging file to a source and target format. 
    For example :
        West NNP B-NP B-MISC
        Indian NNP I-NP I-MISC
        all-rounder NN I-NP O
        Phil NNP I-NP B-PER
        Simmons NNP I-NP I-PER
        . . O O

        is converted two files:
        *.source:
            West Indian all-rounder Phil Simmons .
        and *.target :
            B-MISC I-MISC O B-PER I-PER O

    """

    out_dir = os.path.join(files_dir, "fseq-outputs")

    # create output directory if not exists
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    path = os.path.join(files_dir, split)

    if not os.path.exists(path):
        return None  # file does not exist

    f = open(path)
    data = []
    sentence = []
    label = []

    for line in f:
        if not line.strip() or len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n" or line[0] == '.':
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue

        splits = line.split()
        word, tag = splits[0], splits[-1]
        sentence.append(word.strip())
        label.append(tag.strip())

    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []

    # write output
    outsrc_path = os.path.join(out_dir, '{}.source'.format(split))
    outtgt_path = os.path.join(out_dir, '{}.target'.format(split))
    f_outsrc = open(outsrc_path, 'w')
    f_outtgt = open(outtgt_path, 'w')

    for src, tgt in data:
        assert len(src) == len(tgt)
        f_outsrc.write(' '.join(src) + '\n')
        f_outtgt.write(' '.join(tgt) + '\n')

    logger.info("Wrote source, target to {} {}".format(
        outsrc_path, outtgt_path))
    return os.path.join(out_dir, split)  # return split path prefix


def create_bpe(path_pref, spm_model):
    """
    Takes a split path prefix with .source and .target file, encodes source using BPE, modifies target to tag first token for each word

    For example, the source, target pair: 
    West Indian all-rounder Phil Simmons . -----> B-MISC I-MISC O B-PER I-PER O

    is converted to:
    (We _st In _dian all _- round _er _Ph _il _sim _mons _. -----> B_MISC <pad> I-MISC <pad> O <pad> <pad> <pad> <pad> B-PER <pad> I-PER <pad> O)

    Args:
        path_pref (str): path prefix of this split.
        spm_model: directory to the trained sentencepiece model

    """

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_model)

    def encode(l):
        return sp.EncodeAsPieces(l)

    # input files
    f_source = open(path_pref + '.source', encoding='utf-8')
    f_target = open(path_pref + '.target', encoding='utf-8')

    # bpe out files
    f_source_out = open(path_pref + '.source.bpe', 'w+', encoding='utf-8')
    f_target_out = open(path_pref + '.target.bpe', 'w+', encoding='utf-8')

    for s, t in zip(f_source, f_target):
        cur_source = []
        cur_target = []
        for word, tag in zip(s.split(), t.split()):
            enc_word = encode(word)
            
            bpe_tags = [tag] + ['<pad>'] * (len(enc_word) - 1)
            cur_source.extend(enc_word)
            cur_target.extend(bpe_tags)

        ## make sure 
        assert len(cur_source) == len(cur_target)
        
        # write bpe to files
        f_source_out.write(' '.join(cur_source) + '\n')
        f_target_out.write(' '.join(cur_target) + '\n')


def main(args):
    utils.import_user_module(args)
    os.makedirs(args.destdir, exist_ok=True)

    logger.addHandler(logging.FileHandler(
        filename=os.path.join(args.destdir, 'preprocess.log'),
    ))
    logger.info(args)

    task = tasks.get_task(args.task)

    # TODO: check base if task == seqtag
    assert args.seqtag_data_dir is not None, "you must provide directory for original sequence tagging data"

    args.trainpref = convert_seqtag_format_to_src_tgt(
        args.seqtag_data_dir, split='train')
    args.validpref = convert_seqtag_format_to_src_tgt(
        args.seqtag_data_dir, split='valid')
    args.testpref = convert_seqtag_format_to_src_tgt(
        args.seqtag_data_dir, split='test')

    args.source_lang = 'source'
    args.target_lang = 'target'

    if args.bpe:
        logger.info("BPEing data...")
        assert args.bpe in [
            'sentencepiece'], "not supported BPE method :{}".format(args.bpe)
        assert args.bpe != 'sentencepiece' or args.sentencepiece_model is not None, "you must specify directory for the sentencepiece model"

        # encode bpe
        create_bpe(args.trainpref, args.sentencepiece_model)
        create_bpe(args.validpref, args.sentencepiece_model)
        create_bpe(args.testpref, args.sentencepiece_model)

        args.source_lang = 'source.bpe'
        args.target_lang = 'target.bpe'

    def train_path(lang):
        return "{}{}".format(args.trainpref, ("." + lang) if lang else "")

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):
        return dest_path("dict", lang) + ".txt"

    def build_dictionary(filenames, src=False, tgt=False):
        assert src ^ tgt
        return task.build_dictionary(
            filenames,
            workers=args.workers,
            threshold=args.thresholdsrc if src else 0,
            nwords=args.nwordssrc if src else -1,
            padding_factor=1,
        )

    target = not args.only_source

    if not args.srcdict and os.path.exists(dict_path(args.source_lang)):
        raise FileExistsError(dict_path(args.source_lang))
    if target and not args.tgtdict and os.path.exists(dict_path(args.target_lang)):
        raise FileExistsError(dict_path(args.target_lang))

    if args.joined_dictionary:
        assert not args.srcdict or not args.tgtdict, \
            "cannot use both --srcdict and --tgtdict with --joined-dictionary"

        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        elif args.tgtdict:
            src_dict = task.load_dictionary(args.tgtdict)
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary(
                {train_path(lang) for lang in [args.source_lang, args.target_lang]}, src=True
            )
        tgt_dict = src_dict
    else:
        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary(
                [train_path(args.source_lang)], src=True)

        if target:
            if args.tgtdict:
                tgt_dict = task.load_dictionary(args.tgtdict)
            else:
                assert args.trainpref, "--trainpref must be set if --tgtdict is not specified"
                tgt_dict = build_dictionary(
                    [train_path(args.target_lang)], tgt=True)
        else:
            tgt_dict = None

    src_dict.save(dict_path(args.source_lang))
    if target and tgt_dict is not None:
        tgt_dict.save(dict_path(args.target_lang))

    def make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers):
        logger.info("[{}] Dictionary: {} types".format(lang, len(vocab) - 1))
        n_seq_tok = [0, 0]
        replaced = Counter()

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        input_file = "{}{}".format(
            input_prefix, ("." + lang) if lang is not None else ""
        )
        offsets = Binarizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize,
                    (
                        args,
                        input_file,
                        vocab,
                        prefix,
                        lang,
                        offsets[worker_id],
                        offsets[worker_id + 1]
                    ),
                    callback=merge_result
                )
            pool.close()

        ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, lang, "bin"),
                                          impl=args.dataset_impl, vocab_size=len(vocab))
        merge_result(
            Binarizer.binarize(
                input_file, vocab, lambda t: ds.add_item(t),
                offset=0, end=offsets[1]
            )
        )
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, lang)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))

        logger.info(
            "[{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                lang,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
                100 * sum(replaced.values()) / n_seq_tok[1],
                vocab.unk_word,
            )
        )

    def make_binary_alignment_dataset(input_prefix, output_prefix, num_workers):
        nseq = [0]

        def merge_result(worker_result):
            nseq[0] += worker_result['nseq']

        input_file = input_prefix
        offsets = Binarizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize_alignments,
                    (
                        args,
                        input_file,
                        utils.parse_alignment,
                        prefix,
                        offsets[worker_id],
                        offsets[worker_id + 1]
                    ),
                    callback=merge_result
                )
            pool.close()

        ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, None, "bin"),
                                          impl=args.dataset_impl)

        merge_result(
            Binarizer.binarize_alignments(
                input_file, utils.parse_alignment, lambda t: ds.add_item(t),
                offset=0, end=offsets[1]
            )
        )
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, None)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(dataset_dest_file(args, output_prefix, None, "idx"))

        logger.info(
            "[alignments] {}: parsed {} alignments".format(
                input_file,
                nseq[0]
            )
        )

    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1):
        if args.dataset_impl == "raw":
            # Copy original text file to destination folder
            output_text_file = dest_path(
                output_prefix +
                ".{}-{}".format(args.source_lang, args.target_lang),
                lang,
            )
            shutil.copyfile(file_name(input_prefix, lang), output_text_file)
        else:
            make_binary_dataset(vocab, input_prefix,
                                output_prefix, lang, num_workers)

    def make_all(lang, vocab):
        if args.trainpref:
            make_dataset(vocab, args.trainpref, "train",
                         lang, num_workers=args.workers)
        if args.validpref:
            for k, validpref in enumerate(args.validpref.split(",")):
                outprefix = "valid{}".format(k) if k > 0 else "valid"
                make_dataset(vocab, validpref, outprefix,
                             lang, num_workers=args.workers)
        if args.testpref:
            for k, testpref in enumerate(args.testpref.split(",")):
                outprefix = "test{}".format(k) if k > 0 else "test"
                make_dataset(vocab, testpref, outprefix,
                             lang, num_workers=args.workers)

    def make_all_alignments():
        if args.trainpref and os.path.exists(args.trainpref + "." + args.align_suffix):
            make_binary_alignment_dataset(
                args.trainpref + "." + args.align_suffix, "train.align", num_workers=args.workers)
        if args.validpref and os.path.exists(args.validpref + "." + args.align_suffix):
            make_binary_alignment_dataset(
                args.validpref + "." + args.align_suffix, "valid.align", num_workers=args.workers)
        if args.testpref and os.path.exists(args.testpref + "." + args.align_suffix):
            make_binary_alignment_dataset(
                args.testpref + "." + args.align_suffix, "test.align", num_workers=args.workers)

    make_all(args.source_lang, src_dict)
    if target:
        make_all(args.target_lang, tgt_dict)
    if args.align_suffix:
        make_all_alignments()

    logger.info("Wrote preprocessed data to {}".format(args.destdir))

    if args.alignfile:
        assert args.trainpref, "--trainpref must be set if --alignfile is specified"
        src_file_name = train_path(args.source_lang)
        tgt_file_name = train_path(args.target_lang)
        freq_map = {}
        with open(args.alignfile, "r", encoding='utf-8') as align_file:
            with open(src_file_name, "r", encoding='utf-8') as src_file:
                with open(tgt_file_name, "r", encoding='utf-8') as tgt_file:
                    for a, s, t in zip_longest(align_file, src_file, tgt_file):
                        si = src_dict.encode_line(s, add_if_not_exist=False)
                        ti = tgt_dict.encode_line(t, add_if_not_exist=False)
                        ai = list(
                            map(lambda x: tuple(x.split("-")), a.split()))
                        for sai, tai in ai:
                            srcidx = si[int(sai)]
                            tgtidx = ti[int(tai)]
                            if srcidx != src_dict.unk() and tgtidx != tgt_dict.unk():
                                assert srcidx != src_dict.pad()
                                assert srcidx != src_dict.eos()
                                assert tgtidx != tgt_dict.pad()
                                assert tgtidx != tgt_dict.eos()

                                if srcidx not in freq_map:
                                    freq_map[srcidx] = {}
                                if tgtidx not in freq_map[srcidx]:
                                    freq_map[srcidx][tgtidx] = 1
                                else:
                                    freq_map[srcidx][tgtidx] += 1

        align_dict = {}
        for srcidx in freq_map.keys():
            align_dict[srcidx] = max(
                freq_map[srcidx], key=freq_map[srcidx].get)

        with open(
                os.path.join(
                    args.destdir,
                    "alignment.{}-{}.txt".format(args.source_lang,
                                                 args.target_lang),
                ),
                "w", encoding='utf-8'
        ) as f:
            for k, v in align_dict.items():
                print("{} {}".format(src_dict[k], tgt_dict[v]), file=f)


def binarize(args, filename, vocab, output_prefix, lang, offset, end, append_eos=False):
    ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, lang, "bin"),
                                      impl=args.dataset_impl, vocab_size=len(vocab))

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(filename, vocab, consumer, append_eos=append_eos,
                             offset=offset, end=end)

    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    return res


def binarize_alignments(args, filename, parse_alignment, output_prefix, offset, end):
    ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, None, "bin"),
                                      impl=args.dataset_impl, vocab_size=None)

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize_alignments(filename, parse_alignment, consumer, offset=offset,
                                        end=end)
    ds.finalize(dataset_dest_file(args, output_prefix, None, "idx"))
    return res


def dataset_dest_prefix(args, output_prefix, lang):
    base = "{}/{}".format(args.destdir, output_prefix)
    if lang is not None:
        lang_part = ".{}-{}.{}".format(args.source_lang,
                                       args.target_lang, lang)
    elif args.only_source:
        lang_part = ""
    else:
        lang_part = ".{}-{}".format(args.source_lang, args.target_lang)

    return "{}{}".format(base, lang_part)


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return "{}.{}".format(base, extension)


def get_offsets(input_file, num_workers):
    return Binarizer.find_offsets(input_file, num_workers)


def cli_main():
    parser = options.get_preprocessing_parser()
    parser.add_argument('--seqtag-data-dir', default=None,
                        type=str, help='directory for IOB formatted data')
    parser.add_argument('--sentencepiece-model', type=str,
                        default=None, help='directorty for the sentencepiece model')

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
