import json
import os
from argparse import ArgumentParser
from pathlib import Path
from nsp.dataset_readers.seq2seq_with_copy_reader import SequenceToSequenceModelWithCopyReader

from nsp.metrics.denotation_accuracy import DenotationAccuracy
from nsp.metrics.denotation_accuracy_proxy import DenotationAccuracyProxy

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--decode-file', type=Path, required=True)
    arg_parser.add_argument('--evaluator', type=str, default='server')
    arg_parser.add_argument('--all-hyps', action='store_true')

    args = arg_parser.parse_args()
    # dataset_file = Path('data/test_scholar.json')
    pred_file = args.decode_file
    # pred_file = Path('data/test_scholar_nat_overnight_a2f9551a_6_template_not_add_canonical_var_to_vocab.json')
    # pred_file = Path('data/debug.jsonl')
    # dataset_reader = SequenceToSequenceModelWithCopyReader(
    #     pretrained_encoder_name='bert-base-uncased'
    # )

    # dataset = dataset_reader.read(dataset_file)
    decode_results = [
        json.loads(line) for line in
        pred_file.open()
    ]

    all_hyps = []
    all_targets = []
    variables = []
    indices = []
    for decode_result in decode_results:
        # assert instance['metadata']['source_tokens'] == decode_result['metadata']['source_tokens']
        target = decode_result['metadata']['target_tokens']

        if args.all_hyps:
            hyps = [
                hyp['tokens']
                for hyp
                in decode_result['predictions']
            ]

        else:
            hyps = [decode_result['predictions'][0]['tokens']]

        if hyps and hyps[0] == target:
            print(hyps[0])

        all_hyps.append(hyps)
        all_targets.append(target)
        variables.append(decode_result['metadata']['variables'])
        indices.append(decode_result['metadata']['index'])

    denot_metric = (
        DenotationAccuracyProxy(os.environ.get('EVALUATION_SERVER_ADDR', 'http://localhost:8081/'))
        if args.evaluator == 'server'
        else DenotationAccuracy()
    )
    denot_metric(all_hyps, all_targets, variables, indices)
    result = denot_metric.get_metric(reset=True)
    print(result)
    json.dump(result, pred_file.with_suffix('.eval_result').open('w'))
