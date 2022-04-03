import json
import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict
import numpy as np

from paraphrase.paraphrase_tree import ParaphraseTree, Example, Node


def load_jsonl(file_path):
    return [
        json.loads(line)
        for line
        in open(file_path)
    ]


def get_entry(idx: str, tree: Dict):
    if 'paraphrase' in idx:
        original_src_idx, _, paraphrase_id_str = idx.partition('-')
        if paraphrase_id_str.count('paraphrase') == 1:
            paraphrase_idx = re.match(r'paraphrase-(\d+)', paraphrase_id_str).group(1)
            return tree[original_src_idx]['children'][paraphrase_idx]
        else:
            m = re.match(r'paraphrase-(\d+)-paraphrase-(\d+)', paraphrase_id_str)
            parent_idx = m.group(1)
            paraphrase_idx = m.group(2)
            node = tree[original_src_idx]['children'][parent_idx]['children'][paraphrase_idx]

            return node
    else:
        return tree[idx]


def generate_html(paraphrase_tree: Dict, output_file: Path, show_all: bool = False):
    examples_html = []

    # all_idx = list(paraphrase_tree)
    # np.random.seed(1234)
    # np.random.shuffle(all_idx)
    # sampled_idx_list = set(all_idx[:1000])
    # paraphrase_tree = {
    #     k: v
    #     for k, v
    #     in paraphrase_tree.items()
    #     if k in sampled_idx_list
    # }

    pi_model_better_than_cur_approach = []
    pi_model_winning_examples = []
    pi_model_failed_examples = []

    def log_pi_model(paraphrase_example_: Dict, parent_example_: Dict):
        if 'is_valid_paraphrase' in paraphrase_example_:
            is_valid_paraphrase = paraphrase_example_['is_valid_paraphrase']
            if is_valid_paraphrase and not paraphrase_example_['is_accepted_by_parser']:
                pi_model_winning_examples.append(paraphrase_example_)
                if 1 not in paraphrase_example_.get('used_in_training_iter', []):
                    pi_model_better_than_cur_approach.append(paraphrase_example_)
            elif not is_valid_paraphrase and (paraphrase_example_['is_accepted_by_parser'] or 1 in paraphrase_example_.get('used_in_training_iter', [])):
                pi_model_failed_examples.append(paraphrase_example_)

    def get_status_html(example_: Dict):
        accepted = example_['is_accepted_by_parser']
        status_str = ''
        if accepted:
            status_str += '<span style="color:green;">Accepted</span>'

        used_in_iter = example_.get('used_in_training_iter')
        if used_in_iter:
            status_str += f'<span style="color:green;padding-left:5px; ">used in: {used_in_iter}</span>'

        if 'is_valid_paraphrase' in example_:
            is_valid_paraphrase = example_['is_valid_paraphrase']
            text_color = 'green' if is_valid_paraphrase else 'black'
            status_str += f'<span style="padding-left:5px;">is_valid_paraphrase: <span style="color:{text_color}">{is_valid_paraphrase}</span></span>'

        status_str = '<span>' + status_str + '</span>'

        return status_str

    for idx, example in paraphrase_tree.items():
        example_html = f"""
<li>
  <p id="e{idx}" onclick="$('#e{idx}-children').toggle();">> {example['can']} <span>({len(example['children'])} children)</span></p>
  <ul id="e{idx}-children" {"" if show_all else "hidden"}>
    {{level1_children}}
  </ul>
</li>     
"""
        level1_children_html = ''
        for paraphrase_example_idx, paraphrase_example in example['children'].items():
            dom_idx = f'{idx}-{paraphrase_example_idx}'
            level1_child_html = f"""
<li>
  <p id="e{dom_idx}" onclick="$('#e{dom_idx}-children').toggle();">> {paraphrase_example['can']} <span>({get_status_html(paraphrase_example)} {len(paraphrase_example['children'])} children)</span></p>
  <ul id="e{dom_idx}-children" {"hidden"}>
    {{level2_children}}
  </ul>
</li>
"""
            log_pi_model(paraphrase_example, example)

            level2_children_html = ''
            for paraphrase_example_l2_idx, paraphrase_example_l2 in paraphrase_example['children'].items():
                dom_idx = f'{idx}-{paraphrase_example_idx}-{paraphrase_example_l2_idx}'
                level2_child_html = f"""
<li>
  <p id="e{dom_idx}">> {paraphrase_example_l2['can']}  <span>({get_status_html(paraphrase_example_l2)})</span></p>
</li>    
"""
                log_pi_model(paraphrase_example_l2, paraphrase_example)
                level2_children_html += level2_child_html

            level1_child_html = level1_child_html.format(level2_children=level2_children_html)
            level1_children_html += level1_child_html

        example_html = example_html.format(level1_children=level1_children_html)
        examples_html.append(example_html)

    examples_html = '\n'.join(examples_html)

    html = f"""
<html>
    <head>
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    </head>
    <body>
        <ul>
            <li>Paraphrase Id. Model Winning: {len(pi_model_winning_examples)}</li>
            <li>Paraphrase Id. Model Better Than Current Filtering Model: {len(pi_model_better_than_cur_approach)}</li>
            <li>Paraphrase Id. Model Losing: {len(pi_model_failed_examples)}</li>
        </ul>
        <br />
        <ul>
            {examples_html}
        </ul>
    </body>
</html>
"""
    with output_file.open('w') as f:
        f.write(html)

    return html


def visualize_tree(
    paraphrase_tree: ParaphraseTree,
    output_file: Path
):
    valid_examples_not_accepted_by_parser = []

    def get_status_html(example_: Dict, level: int):
        children = paraphrase_tree.get_children_examples_by_idx(example_['idx'])
        children_num = len(children)

        html = f'Id: {example_["idx"]}; {children_num} children; '
        if level > 0:
            accepted = example_['is_accepted_by_parser']
            if accepted:
                html += '<span style="color:green;">Accepted;</span>'

            if 'accepted_by_parser_in_iter' in example_:
                iter_idx_list = sorted(example_['accepted_by_parser_in_iter'])
                list_html = ''
                for iter_idx in iter_idx_list:
                    val = example_['accepted_by_parser_in_iter'][iter_idx]
                    text_color = 'green' if bool(val) else 'red'
                    list_html += f'<span style="padding-left:1.5px;color:{text_color}">{iter_idx}:{val}</span>'

                list_html = f'<span style="padding-left:5px">Accepted in: {list_html} ||| </span>'
                html += list_html

            if 'paraphrase_identification_labels' in example_:
                iter_idx_list = sorted(example_['paraphrase_identification_labels'])
                list_html = ''
                for iter_idx in iter_idx_list:
                    val = example_['paraphrase_identification_labels'][iter_idx]
                    text_color = 'green' if bool(val) else 'red'
                    list_html += f'<span style="padding-left:1.5px;color:{text_color}">{iter_idx}:{val}</span>'

                list_html = f'<span style="padding-left:5px">P-ID Label: {list_html} ||| </span>'
                html += list_html

            if 'pruner_metadata' in example_:
                pruner_metadata = example_['pruner_metadata']
                iter_idx_list = sorted(pruner_metadata)
                list_html = ''
                for iter_idx in iter_idx_list:
                    is_false_positive = pruner_metadata[iter_idx].get('is_false_positive', False)
                    if is_false_positive:
                        tag = ''
                        if pruner_metadata[iter_idx].get('pruned_by_parent'):
                            tag = '(by_parent)'
                        list_html += f'<span style="padding-left:1.5px;color:red">Pruned@{iter_idx}{tag}</span>'

                list_html = f'<span style="padding-left:5px">Pruner: {list_html} ||| </span>'
                html += list_html

            if 'pid_model_metadata' in example_:
                pid_model_metadata = example_['pid_model_metadata']
                list_html = ''
                def _get_color(_entry):
                    return 'red' if ':F' in _entry else 'green'
                for entry in pid_model_metadata:
                    list_html += f'<span style="padding-left:1.5px;color:{_get_color(entry)}">{entry}</span>'

                list_html = f'<span style="padding-left:5px">PI Training Data: {list_html} ||| </span>'
                html += list_html

            if 'is_valid_paraphrase' in example_:
                is_valid_paraphrase = example_['is_valid_paraphrase']
                text_color = 'green' if is_valid_paraphrase else 'red'
                html += f'<span style="padding-left:5px;">is_valid_paraphrase: <span style="color:{text_color}">{is_valid_paraphrase}</span></span>'

                if is_valid_paraphrase and not accepted:
                    valid_examples_not_accepted_by_parser.append(example_)

            html = '<span>(' + html + ')</span>'

        return html

    def _generate(node: Node, level: int) -> str:
        example = node.value
        example_idx = str(example['idx'])
        div_id = f'e-{example_idx}'
        node_p = f"""<p id="{div_id}" onclick="$('#{div_id}-children').toggle();">NL: {example['nl']} | {get_status_html(example, level=level)}</p>"""
        child_node_htmls = []
        if node.children:
            for child_node in node.children:
                child_html = _generate(child_node, level + 1)
                child_node_htmls.append(child_html)

        children_node_html = (
            # '<ul>\n' +
            '\n'.join([
                f'<li>\n{child_html}\n</li>'
                for child_html
                in child_node_htmls
            ]) # +
            # '</ul>\n'
        )

        node_html = f"""
<div>
  {node_p}
  <ul id="{div_id}-children" hidden>
    {children_node_html}
  </ul>
</div>
"""
        return node_html

    example_htmls = []
    for node in paraphrase_tree.get_nodes_by_level(0):
        example_html = _generate(node, 0)
        example_htmls.append(example_html)

    examples_html = '\n'.join(
        '<li>\n' + example_html + '\n</li>'
        for example_html
        in example_htmls
    )

    examples_html = f'<ul>\n{examples_html}\n</ul>'
    page_html = f"""
<html>
    <head>
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    </head>
    <body>
        {examples_html}
    </body>
</html>
"""
    with output_file.open('w') as f:
        f.write(page_html)

    np.random.shuffle(valid_examples_not_accepted_by_parser)

    with output_file.with_suffix('.valid_but_not_accepted.txt').open('w') as f:
        for example in valid_examples_not_accepted_by_parser[:500]:
            root_idx = paraphrase_tree.get_root_idx(example["idx"])
            root_nl = paraphrase_tree.get_example_by_id(root_idx)['nl']
            f.write('-' * 10 + '\n')
            f.write(f'Idx: {example["idx"]}\n')
            f.write(f'Utterance: {example["nl"]}\n')
            f.write(f'Original: {root_nl}\n')

    return page_html


def visualize_pruner_example(example: Example, pruner_metadata: Dict, paraphrase_tree: ParaphraseTree) -> str:
    from paraphrase.paraphrase_pruner import normalize_lf
    parser_pruner = 'nbr_log_probs' in pruner_metadata

    output_string = ''
    # pruner_metadata = example['pruner_metadata']
    if 'pruned_by_parent' in pruner_metadata:
        output_string += 'This example is pruned because its parent is pruned/invalid\n'
    elif parser_pruner:
        output_string += f'p(y|x)={pruner_metadata["example_log_prob"]:.3f}\n'
        for i, nbr_idx in enumerate(pruner_metadata['nbr_idx']):
            nbr_example = paraphrase_tree.get_example_by_id(nbr_idx)
            output_string += f'[Nbr {nbr_idx}] {nbr_example["nl"]} ||| ' \
                             f'p(y|x)={pruner_metadata["nbr_log_probs"][i]:.3f} ||| ' \
                             f'triggered: {pruner_metadata["example_log_prob"] < pruner_metadata["nbr_log_probs"][i]} ||| ' \
                             f'{normalize_lf(nbr_example["canonical_lf"])}\n'
    else:
        output_string += f'p_paraphrase(x_p, x)={pruner_metadata["p_x_parent_and_x"]:.3f}\n'
        for i, nbr_idx in enumerate(pruner_metadata['nbr_idx']):
            nbr_example = paraphrase_tree.get_example_by_id(nbr_idx)
            output_string += f'[Nbr {nbr_idx}] {nbr_example["nl"]} ||| Parent:{paraphrase_tree.get_parent_example_by_idx(nbr_example["idx"])["nl"]} ||| ' \
                             f'p(x_nbr_p, x)={pruner_metadata["p_nbr_parent_and_x"][i]:.3f} ||| ' \
                             f'triggered: {pruner_metadata["p_x_parent_and_x"] < pruner_metadata["p_nbr_parent_and_x"][i]} ||| ' \
                             f' {normalize_lf(nbr_example["canonical_lf"])}' \
                             f'\n'

    return output_string


def visualize_example(example: Example, paraphrase_tree: ParaphraseTree) -> str:
    from paraphrase.paraphrase_pruner import normalize_lf

    output_string = f'Example {example["idx"]}:\n'
    output_string += f'[Source] {example["nl"]}\n'
    output_string += f'[Target] {example["lf"]}\n'
    output_string += f'[Normalized Target] {normalize_lf(example["canonical_lf"])}\n'

    parent_example = paraphrase_tree.get_parent_example_by_idx(example['idx'])
    if parent_example:
        output_string += f'[Parent] {parent_example["nl"]}\n'

    return output_string


def analyze_paraphrase_tree(paraphrase_tree: ParaphraseTree, output_file: Path):
    max_depth = paraphrase_tree.depth
    output_string = ''
    for level in range(1, max_depth):
        examples = paraphrase_tree.get_examples_by_level(level)
        example: Example
        for example in examples:
            pruner_metadata = example.get('pruner_metadata', None)

            if pruner_metadata:
                if any(x in pruner_metadata for x in {1, 2, 3, 4, '1', '2', '3', '4'}):
                    example_string = visualize_example(example, paraphrase_tree)
                    for iter_idx in pruner_metadata:
                        int(iter_idx)
                        if pruner_metadata[iter_idx].get('is_false_positive'):
                            example_string += f'**** Iter {iter_idx} ****\n'
                            example_string += visualize_pruner_example(example, pruner_metadata[iter_idx], paraphrase_tree)
                # elif pruner_metadata.get('is_false_positive'):
                #     example_string += visualize_pruner_example(example, pruner_metadata, paraphrase_tree)

                    output_string += example_string
                    output_string += '-' * 10 + '\n'

    with output_file.open('w') as f:
        f.write(output_string)


def main(run_path: Path, end_iter: int, **kwargs):
    assert run_path.exists()

    start_iter_idx = 0
    end_iter_idx = end_iter

    paraphrase_tree = dict()

    for iter_idx in range(start_iter_idx, end_iter_idx):
        iter_dir = run_path / f'round1_iter{iter_idx}'
        if iter_idx == 0 and kwargs.get('iter0_dir'):
            iter_dir = kwargs['iter0_dir']

        if not iter_dir.exists():
            print(f'{iter_dir} does not exists, continue.')
            continue

        pi_model_predictions = None
        pi_model_suffix = 'pi_None_lr2e-05_ep2_ngsimTrue'
        pi_model_name = f'{pi_model_suffix}_model'
        if (iter_dir / pi_model_name).exists():
            print('Loading paraphrase identification model predictions...')
            from paraphrase.paraphrase_identification import load_paraphrase_identification_model_prediction
            pi_train_samples_file = iter_dir / f'train.paraphrased.iter{iter_idx}.{pi_model_suffix}.all.csv'

            if pi_train_samples_file.exists():
                print(f'Prediction samples file name: {pi_train_samples_file}')

                pi_model_predictions = load_paraphrase_identification_model_prediction(
                    pi_train_samples_file,
                    iter_dir / f'{pi_model_suffix}_model' / 'test_results_None.txt'
                )
            else:
                pi_train_samples_file = iter_dir / f'pi_data.{pi_model_suffix}.csv'
                if pi_train_samples_file.exists():
                    pi_model_predictions = load_paraphrase_identification_model_prediction(
                        pi_train_samples_file,
                        iter_dir / f'{pi_model_suffix}_model' / 'test_results_None.txt'
                    )
                else:
                    pi_predictions_file = iter_dir / f'{pi_model_suffix}.to_infer.predictions.tsv'
                    print(f'Prediction file name: {pi_predictions_file}')
                    if pi_predictions_file.exists():
                        pi_model_predictions = {}
                        for line in pi_predictions_file.open():
                            data = line.strip().split('\t')
                            idx = data[0]
                            label = bool(int(data[1]))
                            pi_model_predictions[idx] = label
                    else:
                        raise RuntimeError()

        if iter_idx == 0:
            # examples = load_jsonl(iter_dir / 'all_derives_scholar_6.24985d17.template.k1000.template_split.train.iter0.jsonl')
            seed_train_file = list(iter_dir.glob('all_derives_*train.iter0.jsonl'))[0]
            examples = load_jsonl(seed_train_file)
            # examples = load_jsonl(Path('runs/iterative_learning_run_b92891/round1_iter0') / 'all_derives_scholar_6.24985d17.template.k1000.template_split.train.iter0.jsonl')
            for example in examples:
                example_idx = str(example['idx'])
                paraphrase_tree[example_idx] = dict(can=example['nl'], raw=example['can'], children=dict())
        else:
            paraphrased_examples = load_jsonl(iter_dir / f'train.paraphrased.iter{iter_idx}.jsonl')
            for example in paraphrased_examples:
                original_src_idx, _, paraphrase_id_str = example['idx'].partition('-')
                assert original_src_idx in paraphrase_tree
                root = paraphrase_tree[original_src_idx]
                if iter_idx == 1:
                    paraphrase_idx = re.match(r'paraphrase-(\d+)', paraphrase_id_str).group(1)
                    parent = root
                elif iter_idx == 2:
                    m = re.match(r'paraphrase-(\d+)-paraphrase-(\d+)', paraphrase_id_str)
                    parent_idx = m.group(1)
                    paraphrase_idx = m.group(2)
                    parent = root['children'][parent_idx]
                else:
                    raise ValueError()

                parent['children'].setdefault(
                    paraphrase_idx,
                    dict(
                        can=example['nl'],
                        raw=example['can'],
                        is_accepted_by_parser=example['is_accepted_by_parser'],
                        sim_score=example['sim_score'],
                        children=dict()
                    )
                )

                if pi_model_predictions and 'is_valid_paraphrase' not in example:
                    parent['children'][paraphrase_idx]['is_valid_paraphrase'] = pi_model_predictions[example['idx']]

            parser_input_examples = load_jsonl(iter_dir / f'train.paraphrased.iter{iter_idx}.parser_input.jsonl')
            for example in parser_input_examples:
                example_node: Dict = get_entry(example['idx'], paraphrase_tree)
                example_node.setdefault('used_in_training_iter', list()).append(iter_idx)

            saved_train_paraphrase_file = iter_dir / 'paraphrase_tree.train.jsonl'
            if saved_train_paraphrase_file.exists():
                print(f'Visualizing saved paraphrase tree at {saved_train_paraphrase_file}')
                saved_train_paraphrase_tree = ParaphraseTree.from_jsonl_file(saved_train_paraphrase_file)
                analyze_paraphrase_tree(saved_train_paraphrase_tree, saved_train_paraphrase_file.with_suffix('.visualize.txt'))

    with (run_path / 'paraphrase_tree.json').open('w') as f:
        json.dump(paraphrase_tree, f, indent=2)

    generate_html(paraphrase_tree, run_path / f'paraphrase_tree.{pi_model_suffix}.html', show_all=kwargs['show_all'])


def main_paraphrase_tree(run_path: Path, end_iter: int, paraphrase_tree_file_name: str = 'paraphrase_tree.train.jsonl', **kwargs):
    iter_dir = run_path / f'round1_iter{end_iter}'
    paraphrase_tree_file = iter_dir / paraphrase_tree_file_name

    paraphrase_tree = ParaphraseTree.from_jsonl_file(paraphrase_tree_file)
    visualize_tree(paraphrase_tree, (run_path / paraphrase_tree_file_name).with_suffix('.html'))
    analyze_paraphrase_tree(paraphrase_tree, run_path / 'pruner.visualize.txt')


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('run_path', type=Path)
    arg_parser.add_argument('--iter0_dir', required=False, type=Path)
    arg_parser.add_argument('--show_all', action='store_true')
    arg_parser.add_argument('--end_iter', type=int, default=4)
    arg_parser.add_argument('--paraphrase_tree_file_name', type=str, default='paraphrase_tree.train.jsonl')
    args = arg_parser.parse_args()
    # assert len(sys.argv) == 2, print('Usage: python script.py RUN_DIR')
    # main(**vars(args))
    main_paraphrase_tree(**vars(args))

