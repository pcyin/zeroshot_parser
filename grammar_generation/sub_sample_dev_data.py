import math
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from common.utils import load_jsonl_file, write_jsonl_file
from scipy.special import logsumexp


def sub_sample(
    seed_dataset_path: Path,
    output: Path,
    k: int = 2000,
    strategy: str = 'template',
    lm_score_path: Path = None,
    eta: float = 0.2,
    **kwargs
):
    np.random.seed(1234)

    all_examples = load_jsonl_file(seed_dataset_path)
    selected_examples = []
    if strategy == 'nat_template_sample':
        programs_grouped_by_template = {}
        lm_scores = [
            float(line.strip().split()[1])
            for line
            in lm_score_path.open()
        ]
        for idx, example in enumerate(all_examples):
            lm_score = lm_scores[idx]
            example['can_lm_score'] = lm_score

            programs_grouped_by_template.setdefault(
                example['canonical_lf'], []).append(example)

        can_lfs = []
        p_list = []
        for can_lf, examples in programs_grouped_by_template.items():
            p_lf = float(np.exp(logsumexp([
                e['can_lm_score']
                for e
                in examples
            ])))

            can_lfs.append(can_lf)
            p_list.append(p_lf)

        print(f'num templates: {len(can_lfs)}')
        template_p_naturalness = np.array(p_list)
        template_p_naturalness = template_p_naturalness ** eta
        template_p_naturalness = template_p_naturalness / template_p_naturalness.sum()

        print('Top K most likely templates:')
        print(sorted(template_p_naturalness, reverse=True)[:100])

        idx_list = list(range(len(can_lfs)))
        chosen_templates_idx = np.random.choice(
            idx_list,
            replace=False,
            p=template_p_naturalness,
            size=k
        ) if len(can_lfs) > k else idx_list

        for idx in chosen_templates_idx:
            can_lf = can_lfs[idx]
            examples = programs_grouped_by_template[can_lf]
            p = template_p_naturalness[idx]

            num_example = int(math.ceil(max(1, (k - len(idx_list)) * p)))
            if num_example > 1:
                print(num_example)
            if num_example <= len(examples):
                examples_ = sorted(examples, key=lambda e: e['can_lm_score'], reverse=True)[:num_example]
            else:
                examples_ = examples
                # p_examples = np.exp(np.array([e['can_lm_score'] for e in examples]))
                # p_examples /= p_examples.sum()
                # examples_ = np.random.choice(examples, replace=True, p=p_examples, size=num_example)

            selected_examples.extend(examples_)
    elif strategy == 'nat_iid_sample':
        lm_scores = [
            float(line.strip().split()[1])
            for line
            in lm_score_path.open()
        ]

        p_naturalness = np.exp(np.array(lm_scores))
        p_naturalness = p_naturalness ** eta
        p_naturalness = p_naturalness / p_naturalness.sum()

        print('Top K most likely samples:')
        print(sorted(p_naturalness, reverse=True)[:100])

        selected_examples = np.random.choice(
            all_examples,
            replace=False, p=p_naturalness,
            size=min(k, len(all_examples))
        )
    elif strategy == 'iid_sample':
        selected_examples = np.random.choice(
            all_examples,
            replace=False,
            size=k
        )
    else:
        raise ValueError(strategy)

    for e in selected_examples:
        e['idx'] = str(e['idx']).replace('paraphrase', 'p')

    write_jsonl_file(selected_examples, output)


def collect_examples_from_paraphrase_tree(
    paraphrase_tree_path: Path
):
    from paraphrase.paraphrase_tree import ParaphraseTree

    candidate_examples = []
    paraphrase_tree = ParaphraseTree.from_jsonl_file(paraphrase_tree_path)
    for level in range(1, paraphrase_tree.depth):
        for example in paraphrase_tree.get_examples_by_level(level):
            if example['is_valid_paraphrase']:
                candidate_examples.append(example)

    write_jsonl_file(
        candidate_examples,
        paraphrase_tree_path.with_suffix('.valid_examples.jsonl')
    )


if __name__ == '__main__':
    arg_parser = ArgumentParser('dump dataset')
    arg_parser.add_argument(
        '--action', choices=['collect', 'sample'], required=True
    )
    arg_parser.add_argument('--paraphrase-tree-path', type=Path, required=False)

    arg_parser.add_argument(
        '--strategy', type=str, choices=['nat_template_sample', 'nat_iid_sample', 'template', 'iid_sample'],
        default='by_program_depth'
    )
    arg_parser.add_argument('--seed-dataset-path', type=Path, required=False, default=None)
    arg_parser.add_argument('--output', type=Path, required=False)
    arg_parser.add_argument('--lm-score-path', type=Path, default=None, required=False)
    arg_parser.add_argument('--k', type=int, default=2000)
    arg_parser.add_argument('--eta', type=float, default=0.4)

    args = arg_parser.parse_args()

    if args.action == 'collect':
        collect_examples_from_paraphrase_tree(
            args.paraphrase_tree_path
        )
    else:
        sub_sample(**vars(args))
