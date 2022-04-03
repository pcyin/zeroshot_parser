import json
from pathlib import Path
from common.utils import load_jsonl_file


def main():
    file_name = 'runs/iterative_learning_vanilla_canonical_lf_123619/round1_iter1/all_derives_scholar_6.7dbe4611.norule_total.pruned_lf.k2000.iid_split.lm_threshold5.0.ver_f46264dc.+comp+sup.iter1.jsonl'
    file_path = Path(file_name)
    examples = load_jsonl_file(file_path)

    with file_path.with_suffix('.t5_input.tsv').open('w') as f:
        for e in examples:
            f.write(f'{e["idx"]}\t{e["can"]}\n')


if __name__ == '__main__':
    main()
