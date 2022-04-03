import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm


def write_jsonl_file(examples: List[Dict], file_path: Path) -> None:
    with file_path.open('w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')


def load_jsonl_file(file_path: Path, fast=False) -> List[Dict]:
    examples = []
    if fast:
        import ujson as _json
    else:
        _json = json

    num_lines = len(file_path.open().readlines())
    with file_path.open('r') as f:
        for line in tqdm(f, total=num_lines, desc=f'Loading {file_path}'):
            examples.append(
                _json.loads(line.strip())
            )

    return examples
