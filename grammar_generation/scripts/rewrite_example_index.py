import json, sys
from pathlib import Path

from common.utils import load_jsonl_file


def main(input_file: Path):
    output_file = input_file.with_suffix(f'.norm_idx{input_file.suffix}')
    examples = load_jsonl_file(input_file)

    with output_file.open('w') as f:
        for e in examples:
            e['idx'] = str(e['idx']).replace('paraphrase', 'p')
            f.write(
                json.dumps(e) + '\n'
            )


if __name__ == '__main__':
    input_file = Path(sys.argv[1])
    assert input_file.exists()
    main(input_file)
