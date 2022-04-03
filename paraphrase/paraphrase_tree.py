import json
import re
import time
try:
    from functools import cached_property
except:
    from backports.cached_property import cached_property
from pathlib import Path
from typing import List, Dict, Optional

import spacy
from spacy.lang.en import English

from common.utils import load_jsonl_file
from paraphrase.utils import normalize_utterance


Example = Dict


class Node:
    def __init__(self, value: Dict, children: List['Node'] = None):
        self.value = value
        self.children: List['Node'] = children or list()

    def add_child(self, child_node: 'Node'):
        self.children.append(child_node)


class ParaphraseTree(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._node_index = dict()
        self._node_index_by_level = dict()

    @cached_property
    def _nlp_model(self) -> English:
        return spacy.load("en_core_web_md", disable=['parser', 'entity_linker', 'sentencizer', 'ner', 'textcat'])

    def add_example(self, example_idx, example: Example, parent_idx=None):
        if parent_idx:
            parent = self.get_node_by_id(parent_idx)
            if example_idx in self._node_index:
                node = self._node_index[example_idx]
                node.value = example
            else:
                node = Node(example)
                parent.add_child(node)
        else:
            if example_idx in self:
                node = self[example_idx]
                node.value = example
            else:
                node = Node(example)
                self[example_idx] = node

        if 'normalized_can' not in example:
            example['normalized_can'] = normalize_utterance(self._nlp_model(example['can']))

        if example_idx not in self._node_index:
            level = example_idx.count('paraphrase')
            # As of python 3.7 standard dicts are order-preserving
            self._node_index_by_level.setdefault(level, dict())[example_idx] = node
            self._node_index[example_idx] = node

        return example

    def get_nodes_by_level(self, level: int) -> List[Node]:
        return [
            node
            for idx, node
            in self._node_index_by_level.get(level, dict()).items()
        ]

    def get_examples_by_level(self, level: int) -> List[Example]:
        return [
            node.value
            for node
            in self.get_nodes_by_level(level)
        ]

    def get_node_by_id(self, idx) -> Node:
        return self._node_index[str(idx)]

    def get_example_by_id(self, idx) -> Dict:
        return self.get_node_by_id(idx).value

    @classmethod
    def from_examples(cls, examples: List[Example]) -> 'ParaphraseTree':
        root = ParaphraseTree()
        root.update_with_examples(examples)

        return root

    @classmethod
    def from_jsonl_file(cls, file_path: Path) -> 'ParaphraseTree':
        file_path = Path(file_path)
        examples = load_jsonl_file(file_path)

        # examples_idx = set()
        # cleaned_examples = []
        # for example in examples:
        #     if str(example['idx']) not in examples_idx:
        #         cleaned_examples.append(example)
        #         examples_idx.add(str(example['idx']))

        return cls.from_examples(examples)

    def update_with_examples(self, examples: List[Example]) -> 'ParaphraseTree':
        if examples and 'normalized_can' not in examples[0]:
            sents = [e['can'] for e in examples]
            t1 = time.time()
            sent_docs = self._nlp_model.pipe(sents)
            normalized_cans = [
                normalize_utterance(sent)
                for sent
                in sent_docs
            ]
            t2 = time.time()
            print(f'[ParaphraseTree] {t2 - t1:.2f}s took to normalize utterances...')
            for idx, example in enumerate(examples):
                example['normalized_can'] = normalized_cans[idx]

        for example in examples:
            example_idx = str(example['idx'])
            if 'paraphrase' in example_idx:
                # re.match(r'paraphrase-(\d+)', example_idx).group(1)
                m = re.match(r'^(.*)-paraphrase-(\d+)$', example_idx)
                parent_idx = m.group(1)
                child_idx = int(m.group(2))
                # print(f'parent_idx {parent_idx}, child_id {child_idx}')
                self.add_example(example_idx, example, parent_idx=parent_idx)
            else:
                # root level
                self.add_example(example_idx, example)

        return self

    @property
    def depth(self) -> int:
        # start from 0
        return len(list(self._node_index_by_level.keys()))

    def save_to(self, file_path: Path):
        with file_path.open('w') as f:
            for depth in range(self.depth):
                examples = self.get_examples_by_level(depth)
                examples = sorted(examples, key=lambda e: str(e['idx']))
                for example in examples:
                    f.write(json.dumps(example, default=str) + '\n')

    @staticmethod
    def get_root_idx(idx: str) -> str:
        while '-paraphrase-' in idx:
            idx = ParaphraseTree.get_parent_idx(idx)

        return idx

    @staticmethod
    def get_parent_idx(idx: str) -> str:
        idx = str(idx)
        if 'paraphrase' in idx:
            m = re.match(r'^(.*)-paraphrase-(\d+)$', idx)
            parent_idx = m.group(1)
            return parent_idx
        else:
            return None

    def get_parent_example_by_idx(self, idx: str):
        parent_idx = self.get_parent_idx(idx)
        if parent_idx is not None:
            return self.get_example_by_id(parent_idx)

        return None

    def get_children_examples_by_idx(self, idx: str) -> List[Example]:
        node = self.get_node_by_id(idx)
        children = node.children

        return [
            n.value
            for n
            in children
        ]

    def get_descent_examples(self, idx: str) -> List[Example]:
        root_node = self.get_node_by_id(idx)
        result = []

        def visit(node: Node):
            result.extend(n.value for n in node.children)
            for child_node in node.children:
                visit(child_node)

        visit(root_node)

        return result

    def get_ancester_examples(self, idx: str) -> List[Example]:
        parent_example = self.get_parent_example_by_idx(idx)
        result = []
        while parent_example:
            result.append(parent_example)
            parent_example = self.get_parent_example_by_idx(parent_example['idx'])

        return result
