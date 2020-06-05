#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import logging
import math
import os
import sys

import torch

from fairseq import bleu, checkpoint_utils, options, tasks, utils
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.data import encoders
from seqeval.metrics import classification_report, f1_score

def main(args):

    assert args.path is not None, '--path required for prediction!'

    if args.results_path is not None:
        os.makedirs(args.results_path, exist_ok=True)
        output_path = os.path.join(args.results_path, 'predict-{}.txt'.format(args.pred_subset))
        with open(output_path, 'w', buffering=1) as h:
            return _main(args, h)
    else:
        return _main(args, sys.stdout)

def _main(args, output_file):
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        stream=output_file,
    )
    logger = logging.getLogger('fairseq_cli.predict')

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.pred_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    logger.info('loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.path),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Optimize ensemble for generation TODO
    for model in models:
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.pred_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        default_log_format=('tqdm' if not args.no_progress_bar else 'none'),
    )


    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x


    model = models[0] ## Use first model only, TODO ensembles
    y_true = []
    y_pred = []

    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if 'net_input' not in sample:
            continue

        for i, sample_id in enumerate(sample['id'].tolist()):

            # Remove padding
            src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
            target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

            ### Predict
            _, _, y_true_, y_pred_ = task._predict_with_seqeval(sample, model) # get labels and predictions
            y_true_ = y_true_[0]
            y_pred_ = y_pred_[0]

            y_true.append(y_true_) # append to all labels and predictions
            y_pred.append(y_pred_)
        
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, args.remove_bpe)
            else:
                src_str = ""

            target_str = tgt_dict.string(
                target_tokens,
                args.remove_bpe,
                escape_unk=True,
            )

            src_str = decode_fn(src_str)
            target_str = decode_fn(target_str)
            pred_str = ' '.join(y_pred_)

            if not args.quiet:
                print('S-{}\t{}'.format(sample_id, src_str), file=output_file)
                print('P-{}\t{}'.format(sample_id, pred_str), file=output_file)

        clf_report = classification_report(y_true, y_pred)
        logger.info("Eval results on {} subset: \n\n{} ".format(args.pred_subset, clf_report))


def cli_main():
    parser = options.get_prediction_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
