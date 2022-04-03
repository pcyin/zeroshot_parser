from pathlib import Path
from types import SimpleNamespace
from typing import List

from sh import git

from grammar_generation.generate_dataset_from_sampled_examples import generate


RULE_SETS = {
    # Comparison
    'Comparative': [
        '$ComparativeLtPREP', '$ComparativeGtPREP',  # published before, after
    ],
    'SuperlativeNounAdj': [
        '$RelNPSuperlative', '$RelNPSuperlativeMin',  # highest height, lowest elevation,
        '$RelSubSuperlativeAdj', '$RelSubSuperlativeMinAdj',  # most cited, first
    ],
    'SuperlativeRelation': [
        '$SubjectCountSuperlativeRelationNPPrep',  # the most popular topic for
        '$RelationChainVP', '$RelationChainVP/r', '$RelationChainVP/NP', '$RelationChainVP/NP/r',  # publish mostly in
    ],
    'SuperlativeRelationChain': [
        '$RelationChainVP', '$RelationChainVP/r', '$RelationChainVP/NP', '$RelationChainVP/NP/r',  # publish mostly in
    ],
    'CompoundNoun': [
        '$CompPaperUnaryNP',
    ]
}


ALL_DOMAIN_PRODUCTION_TYPES = [
    (
        None,
        {
            'disabled': None
        }
    ),
    (
        'with_multihop',
        {
            'disabled': RULE_SETS['Comparative'] + RULE_SETS['SuperlativeNounAdj'] + RULE_SETS['SuperlativeRelation'] +
                        RULE_SETS['CompoundNoun']
        }
    ),
    (
        '+comp',
        {
            'disabled': RULE_SETS['SuperlativeNounAdj'] + RULE_SETS['SuperlativeRelation'] + RULE_SETS['CompoundNoun']
        }
    ),
    (
        # This is the final configuration used in SCHOLAR.
        '+comp+sup',
        {
            'disabled': RULE_SETS['CompoundNoun']
        },
    ),
    (
        'no_comp',
        {
            'disabled': RULE_SETS['Comparative']
        },
    ),

    (
        'no_compound_noun',
        {
            'disabled': RULE_SETS['CompoundNoun']
        }
    ),
    (
        'no_sup_relation_chain',
        {
            'disabled': RULE_SETS['SuperlativeRelationChain']
        },
    ),
    (
        'no_compound_noun_no_sup_relation_chain',
        {
            'disabled': RULE_SETS['CompoundNoun'] + RULE_SETS['SuperlativeRelationChain']
        },
    )
]


def main(
    samples_file: Path,
    scores_file: Path,
    git_sha: str,
    strategy: str = 'by_program_depth',
    per_program_depth_sample_strategy: str = 'rank_by_lm_score',
    production_groups: List[str] = None
):
    # Dev files are not used in the paper.
    dev_example_ratio = 0.1
    dev_sets = {}

    production_groups = production_groups or [None]
    domain_disabled_production_types = [
        group
        for group in ALL_DOMAIN_PRODUCTION_TYPES
        if group[0] in production_groups
    ]

    is_ablation = False
    for split_method in ['iid']:  # 'iid', 'template'
        for disabled_production_info in domain_disabled_production_types:
            if len(domain_disabled_production_types) > 1 and disabled_production_info and disabled_production_info[1]['disabled'] and not is_ablation:
                print('Start generating ablation splits....')
                is_ablation = True

            Ks = [500, 1000, 2000, 4000, 8000]

            for k in Ks:
                lm_prob_thresholds = [None]
                if scores_file:
                    lm_prob_thresholds = [5.0]
                # if not is_ablation:
                # lm_prob_thresholds.append(1000.0)

                strategy_name = strategy
                if per_program_depth_sample_strategy:
                    strategy_name += f'_{per_program_depth_sample_strategy}'
                for lm_prob_threshold in lm_prob_thresholds:  # , 10.0, 20.0, 1000.0
                    output_file = samples_file.with_suffix(
                        f'.k{k}.{split_method}_split.{strategy_name}' +
                        (f'.lm_threshold{lm_prob_threshold}' if lm_prob_threshold is not None else '') +
                        f'.ver_{git_sha}.jsonl'
                    )

                    disabled_production_types = None
                    prod_signature = None
                    if disabled_production_info[0]:  # is not None
                        prod_signature = disabled_production_info[0]
                        disabled_production_types = disabled_production_info[1]['disabled']
                        output_file = output_file.with_suffix(f'.{prod_signature}.jsonl')

                    signature = f'{split_method}|||{prod_signature}'
                    print(disabled_production_types)

                    output = generate(
                        SimpleNamespace(
                            samples_file=samples_file,
                            num_samples=None,
                            lm_scores_file=scores_file,
                            dev_ratio=dev_example_ratio,
                            output_file=output_file,
                            strategy=strategy,
                            per_program_depth_sample_strategy=per_program_depth_sample_strategy,
                            # strategy='by_program_depth',
                            split_method=split_method,
                            num_programs_each_depth=k,
                            lm_prob_threshold=lm_prob_threshold,
                            disabled_production_types=disabled_production_types,
                            # cache=not is_ablation
                            cache=False,
                            num_examples=5400
                        ),
                        dev_set=dev_sets.get(signature, None)
                    )

                    if dev_sets.get(signature, None) is None:
                        print(f'Use {output_file} as dev set')
                        dev_sets[signature] = output['dev_examples']


if __name__ == '__main__':
    git_sha = git('rev-parse', '--short', 'HEAD').strip()

    samples_file = Path(
        'grammar_generation/data/'
        'all_derives_scholar_6.pruned_lf.txt'
    )

    assert samples_file.exists()
    lm_scores_file = samples_file.with_suffix('.lm_score.txt')
    assert lm_scores_file.exists()

    main(
        samples_file,
        scores_file=None,
        git_sha=git_sha,
        strategy='random',
        production_groups=['+comp+sup']
    )

    samples_file = Path(
        'grammar_generation/data/'
        'all_derives_geo_6.pruned_lf.txt'
    )

    assert samples_file.exists()
    lm_scores_file = samples_file.with_suffix('.lm_score.txt')
    assert lm_scores_file.exists()

    main(
        samples_file,
        scores_file=None,
        git_sha=git_sha,
        strategy='random',
    )
