import json
from pathlib import Path
from typing import List, Union, Tuple, TypeVar, Dict, Optional
from pprint import PrettyPrinter
import pyparsing
import sexpdata
from tqdm import tqdm

from common.utils import load_jsonl_file

Sexp = Union[str, List['Sexp']]
T = TypeVar('T')


transformation_rules = [
    (
        'keyphrase_author_keyphrase',
        [
            {'rel': 'write_author_paper', 'type': ('_author_', '_paper_')},
            {'rel': 'keyphrase_paper_keyphrase', 'type': ('_paper_', '_keyphrase_')}
        ]
    ),
    (
        'cites_author_author',
        [
            {'rel': 'write_author_paper', 'type': ('_author_', '_paper_')},
            {'rel': 'cites_paper_paper', 'type': ('_paper_', '_paper_')},
            {'rel': '!write_author_paper', 'type': ('_paper_', '_author_')}
        ]
    )
]


def tree_from_rule(rule: Tuple):
    computed_property = rule[0]
    relations = rule[1]

    # @getProperty(@getProperty(argument, rel_1), rel_2)
    get_prop_sexp: Sexp = []
    get_prop_sexp = '$Target$'
    for relation in relations:
        rel_string = relation['rel']
        if rel_string.startswith('!'):
            rel_string_sexp = ['call', '@reverse', ['string', rel_string[1:]]]
        else:
            rel_string_sexp = ['string', rel_string]

        node = ['call', 'SW.getProperty', [get_prop_sexp], rel_string_sexp]
        get_prop_sexp = node

    filter_sexp: Sexp = []


def modify_program_for_normalization(program: Sexp) -> Tuple[Sexp, bool]:
    modified = False

    if isinstance(program, str):
        return program, modified

    op = program[0]
    if op == 'call':
        func = program[1]
        if func == 'SW.filter':
            arg_1 = program[2]
            # if there is a nested filter call
            if arg_1[0] == 'call' and arg_1[1] == 'SW.filter':
                relation = program[3][1]
                child_func_call_relation = arg_1[3][1]

                swap = child_func_call_relation < relation

                if swap:
                    new_child_sexp: Sexp = []
                    for arg_idx, arg in enumerate(program):
                        if arg_idx != 2:
                            new_child_sexp.append(arg)
                        else:
                            new_child_sexp.append(arg_1[2])

                    program = list(arg_1)
                    program[2] = new_child_sexp
                    modified = True

    new_program = []
    for arg_idx, arg in enumerate(program):
        arg, arg_modified = modify_program_for_normalization(arg)
        new_program.append(arg)
        modified |= arg_modified

    return new_program, modified


def clone_sexp(program: Sexp) -> Sexp:
    if isinstance(program, list):
        return [clone_sexp(arg) for arg in program]
    else:
        return program


def replace_program(program: Sexp, source: Sexp, target: Sexp) -> Sexp:
    if program == source:
        return target
    elif isinstance(program, list):
        new_program = []
        for arg in program:
            new_arg = replace_program(arg, source, target)
            new_program.append(new_arg)

        return new_program
    else:
        return program


def normalize_reverse_call(program: Sexp) -> Sexp:
    def visit(node: Sexp) -> Sexp:
        if isinstance(node, str):
            return node

        # ( string !author_paper_author ) -> ( call SW.reverse ( string author_paper_author ) )
        if len(node) == 2 and node[0] == 'string' and isinstance(node[1], str) and node[1].startswith('!'):
            property_name = node[1][1:]
            assert len(property_name) >= 1

            new_node: Sexp = ['call', 'SW.reverse', ['string', property_name]]
        else:
            new_node = [visit(child) for child in node]

        return new_node

    return visit(program)


def normalize_domain_call(program: Sexp) -> Sexp:
    def visit(node: Sexp) -> Sexp:
        if isinstance(node, str):
            return node

        # ( call SW.domain ( string !author_paper_author ) )
        if node[0] == 'call' and len(node) == 3 and node[1] == 'SW.domain':
            property_name: str = node[2] if isinstance(node[2], str) else node[2][1]

            # author_paper_author
            property_type = property_name.split('_')[-2]
            if property_name.startswith('!'):
                property_type = property_name.split('_')[-1]

            property_type = f'fb:en.{property_type}'
            new_node = ['call', 'SW.getProperty', ['call', 'SW.singleton', property_type], ['string', '!', 'type']]
        else:
            new_node = [visit(child) for child in node]

        return new_node

    return visit(program)


def remove_ensure_numeric_property(program: Sexp) -> Sexp:
    def visit(node: Sexp) -> Sexp:
        if isinstance(node, str):
            return node

        op_name = node[0]
        if op_name == 'call' and len(node) == 3 and '.ensureNumericProperty' in node[1]:
            return node[2]
        else:
            new_node = []
            for child_node in node:
                new_child = visit(child_node)
                new_node.append(new_child)

            return new_node

    return visit(program)


def normalize_program(
    program: Sexp,
    normalize_filter_order: bool = True,
    strip_ensure_numeric_property: bool = False
) -> Sexp:
    program = normalize_domain_call(program)
    program = normalize_reverse_call(program)

    if strip_ensure_numeric_property:
        program = remove_ensure_numeric_property(program)

    if normalize_filter_order:
        program = normalize_program_filter_order(program)

    return program


def normalize_program_filter_order(program: Sexp) -> Sexp:
    modified = True

    while modified:
        program, modified = modify_program_for_normalization(program)

    return program


def flatten(nested_list: List[List[T]]) -> List[T]:
    return [
        element
        for sub_list in nested_list
        for element in sub_list
    ]


def sexp_to_tokenized_string(sexp: Sexp) -> List[str]:
    """
    Shamelessly borrowed from data flow
    Generates tokenized string representation from S-expression
    """
    if isinstance(sexp, list):
        return ['('] + flatten([sexp_to_tokenized_string(f) for f in sexp]) + [')']
    else:
        return [sexp]


def parse_sexp_string(sexp_string: str) -> Sexp:
    sexp = sexpdata.loads(sexp_string)

    def _strip(node) -> Sexp:
        if isinstance(node, sexpdata.Symbol):
            return node._val
        elif isinstance(node, (str, int, float)):
            return str(node)
        else:
            assert isinstance(node, list)
            new_node = []
            for arg in node:
                new_node.append(_strip(arg))

            return new_node

    return _strip(sexp)


def normalize_lf(lf_string: str) -> str:
    return ' '.join([
        x
        for x
        in lf_string.strip().split(' ')
        if x not in {'(', ')', 'name'}
    ])


def get_normalized_program_key(program_string: str):
    sexp = parse_sexp_string(program_string)
    normalized_sexp = normalize_program(sexp)
    program_string = ' '.join(sexp_to_tokenized_string(normalized_sexp))

    return normalize_lf(program_string)


def get_normalized_program_to_example_index(examples: List[Dict]) -> Dict:
    index = {}
    for example in examples:
        key = get_normalized_program_key(example['lf'])
        index.setdefault(key, []).append(example)

    return index


def normalize_lf(lf_string: str) -> str:
    return ' '.join([
        x
        for x
        in lf_string.strip().split(' ')
        if x not in {'(', ')', 'name'}
    ])


def normalize_program_string(program_string: str):
    sexp = parse_sexp_string(program_string)
    normalized_sexp = normalize_program(sexp, strip_ensure_numeric_property=True)
    program_string = ' '.join(sexp_to_tokenized_string(normalized_sexp))

    return normalize_lf(program_string)


def lexicalize_program_string(lf: str, variables: Dict):
    from nsp.metrics.denotation_accuracy import lexicalize_entity

    tokens = lf.split(' ')
    lexicalized_tokens = []
    for i, token in enumerate(tokens):
        new_token = lexicalize_entity(variables, token)
        if token.startswith('misc') and 'number' in new_token and i > 0 and tokens[i - 1] != '(':
            new_token = '( ' + new_token + ' )'

        lexicalized_tokens.append(new_token)

    return ' '.join(lexicalized_tokens)


class LogicalFormMigration:
    def __init__(self, annotation_file: Path, canonical_data_file: Optional[Path] = None):
        canonical_nl_to_lf = {}
        normalized_canonical_lf_strings = set()
        normalized_canonical_lf_to_canonical_examples = {}
        if canonical_data_file:
            canonical_examples = load_jsonl_file(canonical_data_file, fast=True)
            canonical_nl_to_lf = {
                e['nl']: e['lf']
                for e in canonical_examples
            }

            for e in tqdm(canonical_examples, desc='parsing canonical file...'):
                lf_norm = normalize_program_string(e['lf'])
                normalized_canonical_lf_strings.add(
                    lf_norm
                )
                normalized_canonical_lf_to_canonical_examples.setdefault(
                    lf_norm, list()
                ).append(e)

        annotated_program_lf_to_new_lf = {}
        annotated_program_lf_to_new_normalized_lf_string = {}
        not_annotated_lf = set()

        for line in annotation_file.open():
            data = line.strip().split('\t')
            nl = data[0]
            old_lf = data[1]

            if len(data[1]) > 0:
                if len(data) >= 3 and not data[2].startswith('x') and len(data[2]) > 0:
                    new_lf = data[2]

                    if new_lf.startswith('('):
                        try:
                            old_lf_normalized = normalize_program_string(old_lf)
                            new_lf_normalized = normalize_program_string(new_lf)
                            annotated_program_lf_to_new_lf[old_lf_normalized] = new_lf
                            annotated_program_lf_to_new_normalized_lf_string[old_lf_normalized] = new_lf_normalized
                        except Exception as e:
                            print(data)
                            raise e
                    else:
                        utterance = new_lf
                        if utterance in canonical_nl_to_lf:
                            new_lf = canonical_nl_to_lf[utterance]
                            print(f'{utterance} -> {new_lf}')

                            old_lf_normalized = normalize_program_string(old_lf)
                            new_lf_normalized = normalize_program_string(new_lf)
                            annotated_program_lf_to_new_lf[old_lf_normalized] = new_lf
                            annotated_program_lf_to_new_normalized_lf_string[old_lf_normalized] = new_lf_normalized
                        else:
                            print(f'Warning: annotated utterance {utterance} does not appear in canonical examples')
                else:
                    old_lf_normalized = normalize_program_string(old_lf)
                    not_annotated_lf.add(old_lf_normalized)
                    print(f'Warning: [Entry not Annotated] {nl}')

        self.annotated_program_lf_to_new_lf = annotated_program_lf_to_new_lf
        self.annotated_program_lf_to_new_normalized_lf_string = annotated_program_lf_to_new_normalized_lf_string
        self.normalized_canonical_lf_strings = normalized_canonical_lf_strings
        self.lf_with_entry_but_not_annotated = not_annotated_lf
        self.normalized_canonical_lf_to_canonical_examples = normalized_canonical_lf_to_canonical_examples

    @staticmethod
    def get_normalized_program_string(lf):
        return normalize_program_string(lf)

    def has_entry(self, lf: str) -> bool:
        if self.covers(lf):
            return True

        normalized_lf_string = normalize_program_string(lf)
        return normalized_lf_string in self.lf_with_entry_but_not_annotated

    def covers(self, lf: str) -> bool:
        normalized_lf_string = normalize_program_string(lf)

        return (
            normalized_lf_string in self.annotated_program_lf_to_new_lf
            or normalized_lf_string in self.normalized_canonical_lf_strings
        )

    def __call__(self, lf: str, ignore_not_covered_error: bool = False, normalize_filter_order: bool = False) -> str:
        if not ignore_not_covered_error:
            assert self.covers(lf)

        lf_normalized_string = normalize_program_string(lf)
        if lf_normalized_string in self.annotated_program_lf_to_new_lf:
            new_lf = self.annotated_program_lf_to_new_lf[lf_normalized_string]
        else:
            new_lf = lf

        new_lf = ' '.join(
            sexp_to_tokenized_string(
                normalize_program(
                    parse_sexp_string(new_lf),
                    normalize_filter_order=normalize_filter_order,
                    strip_ensure_numeric_property=True
                )
            )
        )

        return new_lf

    def find_canonical_example_by_lf(self, lf: str) -> List[Dict]:
        return list(
            self.normalized_canonical_lf_to_canonical_examples[
                self.get_normalized_program_string(self(lf, ignore_not_covered_error=True))
            ]
        )


def test():
    program = """
    ( call SW.listValue ( call SW.superlative ( call SW.getProperty ( call SW.singleton fb:en.river ) ( string ! type ) ) ( string max ) ( call SW.ensureNumericProperty ( string len_river_length ) ) ) )
    """

    program = parse_sexp_string(program)
    normalized_program = normalize_program(program, strip_ensure_numeric_property=True)
    print(' '.join(sexp_to_tokenized_string(normalized_program)))

    parser = pyparsing.nestedExpr('(', ')', ignoreExpr=pyparsing.dblQuotedString.copy())

    program = """
    ( call SW.listValue ( call SW.filter ( call SW.filter ( call SW.filter ( call SW.getProperty ( call SW.singleton ( name fb:en.paper ) ) ( string ! type ) ) ( string venue_paper_venue ) ( string = ) fb:en.venue.nips ) ( string publication_year_paper_number ) ( string = ) ( number 2012 year ) ) ( string author_paper_author ) ( string = ) fb:en.author.tom_mitchell ) )
    """

    program = """
    ( call SW.listValue ( call SW.filter ( call SW.filter ( call SW.filter ( call SW.getProperty ( call SW.singleton fb:en.paper ) ( string ! type ) ) ( string publication_year_paper_number ) ( string < ) ( number 2016 year ) ) ( string title_paper_title ) ( string ! = ) ( call SW.concat fb:en.title.nmt fb:en.title.multivariate_data_analysis ) ) ( string author_paper_author ) ( string = ) fb:en.author.dan_klein ) )
    """

    program = """
    ( call SW.listValue ( call SW.countSuperlative ( call SW.domain ( string !keyphrase_paper_keyphrase ) ) ( string max ) ( string !keyphrase_paper_keyphrase ) ( call SW.superlative ( call SW.getProperty ( call SW.singleton fb:en.paper ) ( string ! type ) ) ( string min ) ( string citation_count_paper_number ) ) ) )
    """

    program2 = """
    ( call SW.listValue ( call SW.countSuperlative ( call SW.getProperty ( call SW.singleton fb:en.author ) ( string ! type ) ) ( string max ) ( call SW.reverse ( string author_paper_author ) ) ( call SW.filter ( call SW.getProperty ( call SW.singleton fb:en.paper ) ( string ! type ) ) ( string keyphrase_paper_keyphrase ) ( string = ) fb:en.keyphrase.deep_learning ) ) )
    """

    program = """
    ( call SW.listValue ( call SW.countSuperlative ( call SW.domain ( string !author_paper_author ) ) ( string max ) ( string !author_paper_author ) ( call SW.filter ( call SW.getProperty ( call SW.singleton fb:en.paper ) ( string ! type ) ) ( string keyphrase_paper_keyphrase ) ( string = ) fb:en.keyphrase.deep_learning ) ) )
    """

    # program = """
    # ( call SW.listValue ( call SW.filter ( call SW.filter ( call SW.filter ( call SW.getProperty ( call SW.singleton fb:en.paper ) ( string ! type ) ) ( string publication_year_paper_number ) ( string = ) ( year0 ) ) ( string venue_paper_venue ) ( string = ) venuename0 ) ( string author_paper_author ) ( string = ) authorname0 ) )
    # """

    # program = parser.parseString(program.strip()).asList()
    program = parse_sexp_string(program)
    normalized_program = normalize_program(program)
    from nsp.metrics.denotation_accuracy import format_lf
    print(format_lf(' '.join(sexp_to_tokenized_string(program))))
    print(format_lf(' '.join(sexp_to_tokenized_string(normalized_program))))

    program2 = parse_sexp_string(program2)
    print(normalized_program)
    print(program2)
    print(normalized_program == program2)

    pp = PrettyPrinter(indent=2)
    pp.pprint(program)
    pp.pprint(normalized_program)


def cli_normalize():
    import sys
    from pathlib import Path
    from common.utils import load_jsonl_file

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    examples = load_jsonl_file(input_file)
    with output_file.open('w') as f:
        for example in examples:
            program = ' '.join(
                sexp_to_tokenized_string(
                    normalize_program(
                        parse_sexp_string(example['lf']),
                        normalize_filter_order=False
                    )
                )
            )

            normalized_program = ' '.join(
                sexp_to_tokenized_string(
                    normalize_program(
                        parse_sexp_string(example['lf']),
                        normalize_filter_order=True
                    )
                )
            )

            example['lf'] = program
            example['canonical_lf'] = normalized_program

            f.write(json.dumps(example) + '\n')


if __name__ == '__main__':
    cli_normalize()
    #test()
