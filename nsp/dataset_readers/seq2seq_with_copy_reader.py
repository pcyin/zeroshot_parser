import json
from typing import List, Dict, Tuple, Iterable, Set, Optional, Union

from overrides import overrides
import logging
import re
from pathlib import Path
from re import Match
import numpy as np

from allennlp.data import Vocabulary
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField, MetadataField, NamespaceSwappingField, SequenceField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, PretrainedTransformerTokenizer, SpacyTokenizer, \
    WhitespaceTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer


logger = logging.getLogger(__name__)


PLACEHOLDER_ENTITY_MAP = {
    'Dan Klein': 'authorname',
    'Tom Mitchell': 'authorname',
    'natural language processing': 'keyphrasename',
    'deep learning': 'keyphrasename',
    'machine learning': 'keyphrasename',
    '2012': 'year',
    '2016': 'year',
    'NIPS': 'venuename',
    'Nature': 'journalname',
    'Wikipedia': 'datasetname',
    'ImageNet': 'datasetname',
    'neural machine translation': 'title',
    'Multivariate Data Analysis': 'title'
}


VARIABLE_PLACE_HOLDERS_INDEX = {}
type_names = set(PLACEHOLDER_ENTITY_MAP.values())
for type_name in type_names:
    for var_idx in range(3):
        VARIABLE_PLACE_HOLDERS_INDEX[f'{type_name}{var_idx}'] = len(VARIABLE_PLACE_HOLDERS_INDEX)


def ensure_token_list(token_list):
    tokens = []
    for token in token_list:
        if not isinstance(token, list):
            tokens.append(token)
        else:
            tokens.extend(ensure_token_list(token))

    return tokens


class MeaningRepresentationField(TextField):
    def __init__(
        self,
        tokens: List[Token],
        token_indexers: Dict[str, TokenIndexer],
        source_span_mentions: List[Tuple[int, int]] = None,
    ):
        super().__init__(tokens, token_indexers)

        self.source_span_mentions = source_span_mentions or {}

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for token_idx, token in enumerate(self.tokens):
            if self.tokens[0].text == START_SYMBOL:
                token_idx = token_idx - 1

            for indexer in self._token_indexers.values():
                indexer.count_vocab_items(token, counter)


@DatasetReader.register('seq2seq_with_copy')
class SequenceToSequenceModelWithCopyReader(DatasetReader):
    def __init__(
        self,
        pretrained_encoder_name: Optional[str] = None,
        add_canonical_variable_names_to_vocabulary: bool = True,
        logical_form_data_field: str = 'lf',
        local_form_data_field: str = None,
        only_use_parser_filtered_example: bool = False,
        use_canonical_example: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._white_space_tokenizer = SpacyTokenizer(split_on_spaces=True)
        self._pretrained_encoder_name = pretrained_encoder_name

        if pretrained_encoder_name:
            self._source_tokenizer = PretrainedTransformerTokenizer(
                pretrained_encoder_name,
                tokenizer_kwargs={'use_fast': False}
            )
            self._source_token_indexers = {
                'tokens': PretrainedTransformerIndexer(
                    pretrained_encoder_name, namespace='source_tokens'),
            }
        else:
            self._source_tokenizer = SpacyTokenizer(split_on_spaces=True)
            self._source_token_indexers = {
                "tokens": SingleIdTokenIndexer(namespace='source_tokens', lowercase_tokens=True)
            }

        self._target_tokenizer = WhitespaceTokenizer()

        self._add_canonical_variable_names_to_vocabulary = add_canonical_variable_names_to_vocabulary
        if add_canonical_variable_names_to_vocabulary:
            self._source_tokenizer.tokenizer.add_tokens(sorted(list(VARIABLE_PLACE_HOLDERS_INDEX)))

        if 'tokens' not in self._source_token_indexers:
            raise ConfigurationError(
                f"{self} expects 'source_token_indexers' to contain "
                "a token indexer called 'tokens'."
            )

        self._target_token_indexer = SingleIdTokenIndexer(namespace='target_tokens')

        self._logical_form_data_field = logical_form_data_field
        self._only_use_parser_filtered_example = only_use_parser_filtered_example
        self._use_canonical_example = use_canonical_example

    @property
    def use_pretrained_encoder(self):
        return self._pretrained_encoder_name is not None

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = Path(file_path)

        split = None
        for split_opt in ['train', 'dev', 'test']:
            if split_opt in file_path.name:
                split = split_opt
                break
        split = split or 'train'

        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s (split: %s)", file_path, split)
            examples = [
                json.loads(line)
                for line
                in data_file
            ]

            if split in ['train', 'dev']:
                for example in examples:
                    target_sequence = example[self._logical_form_data_field]
                    yield self.text_to_instance(example["nl"], target_sequence, example['variables'], example['idx'], split)
            else:
                for line_num, example in enumerate(examples):
                    source_sequence, target_sequence, variables = example["nl"], example['lf'], example[
                        "variables"]
                    if not source_sequence:
                        continue

                    yield self.text_to_instance(source_sequence, target_sequence, variables, line_num, split)

    @overrides
    def text_to_instance(
        self,
        source_sequence: str,
        target_sequence: str = None,
        variables: Dict = None,
        index: str = None,
        split: str = None
    ) -> Instance:
        source_sequence = source_sequence.strip()
        source_tokens_on_white_space: List[Token] = self._white_space_tokenizer.tokenize(source_sequence)
        if self.use_pretrained_encoder:
            source_tokens, source_subtoken_offsets = self._tokenize_source(source_sequence)
        else:
            source_tokens = source_tokens_on_white_space
            source_subtoken_offsets = None

        source_field = TextField(source_tokens, self._source_token_indexers)

        # For each token in the source sentence, we keep track of the matching token
        # in the target sentence (which will be the OOV symbol if there is no match).
        source_to_target_token_idx_map_field = NamespaceSwappingField(
            source_tokens, target_namespace='target_tokens'
        )

        fields_dict = {
            "source_tokens": source_field,
            'source_to_target_token_idx_map': source_to_target_token_idx_map_field,
            "source_token_first_appearing_indices": ArrayField(
                np.array(self._tokens_to_first_appearing_indices(source_tokens))
            )
        }

        meta_dict = {
            'source': source_sequence,
            'source_tokens': [token.text for token in source_tokens],
            'variables': variables,
            'index': index,
            'split': split
        }

        if target_sequence is not None:
            target_tokens = self._target_tokenizer.tokenize(target_sequence)
            target_tokens.insert(0, Token(START_SYMBOL))
            target_tokens.append(Token(END_SYMBOL))

            target_field = MeaningRepresentationField(
                target_tokens, {'tokens': self._target_token_indexer},
            )
            fields_dict["target_tokens"] = target_field

            source_and_target_token_first_appearing_indices = self._tokens_to_first_appearing_indices(
                source_tokens + target_tokens
            )

            source_token_ids = source_and_target_token_first_appearing_indices[: len(source_tokens)]
            fields_dict["source_token_first_appearing_indices"] = ArrayField(np.array(source_token_ids))
            target_token_ids = source_and_target_token_first_appearing_indices[len(source_tokens):]
            fields_dict["target_token_first_appearing_indices"] = ArrayField(np.array(target_token_ids))

            meta_dict['target_tokens'] = [token.text for token in target_tokens[1:-1]]

        fields_dict['metadata'] = MetadataField(meta_dict)

        instance = Instance(fields_dict)

        return instance

    def _tokenize_source(
        self,
        source: Union[str, List[Token], List[str]],
    ):
        if isinstance(source, str):
            source_tokens_on_white_space = self._white_space_tokenizer.tokenize(source)
        elif isinstance(source, list):
            if isinstance(source[0], str):
                source_tokens_on_white_space = source
            else:
                assert isinstance(source[0], Token)
                source_tokens_on_white_space = [tok.text for tok in source]
        else:
            raise ValueError(source)

        source_tokens, source_subtoken_offsets = self._source_tokenizer.intra_word_tokenize(
            [tok.text for tok in source_tokens_on_white_space]
        )

        # convert special token indices to [unusedxxx], therefore keeping the size of embeddings
        if self._add_canonical_variable_names_to_vocabulary:
            token: Token
            for token in source_tokens:
                if token.text in VARIABLE_PLACE_HOLDERS_INDEX:
                    token.text_id = self._source_tokenizer.tokenizer._convert_token_to_id(  # noqa
                        f'[unused{VARIABLE_PLACE_HOLDERS_INDEX[token.text]}]')

        return source_tokens, source_subtoken_offsets

    def _get_subtokens_slice(
        self,
        tokens: List[Token],
        span: Tuple[int, int],
        tokens_offset: List[Optional[Tuple[int, int]]] = None
    ) -> Tuple[List[Token], Tuple[int, int]]:
        span_start, span_end = span
        if tokens_offset:
            subword_start = tokens_offset[span_start][0]
            subword_end = tokens_offset[span_end - 1][1] + 1

            tokens_slice = tokens[subword_start: subword_end]
            subword_span = (subword_start, subword_end)
        else:
            tokens_slice = tokens[span_start: span_end]
            subword_span = span

        return tokens_slice, subword_span

    @staticmethod
    def _tokens_to_first_appearing_indices(tokens: List[Union[Token, str]]) -> List[int]:
        """Convert tokens to first appearing indices in the sentence"""
        token_to_first_appearing_index_map: Dict[str, int] = {}
        out: List[int] = []

        for token in tokens:
            out.append(token_to_first_appearing_index_map.setdefault(
                token.text if isinstance(token, Token) else str(token), len(token_to_first_appearing_index_map))
            )

        return out


def main():
    reader = SequenceToSequenceModelWithCopyReader()
    dataset = reader.read('data/calflow.singleturn.top100.txt')
    vocab = Vocabulary.from_instances(dataset.instances)
    print(vocab)


if __name__ == '__main__':
    main()
