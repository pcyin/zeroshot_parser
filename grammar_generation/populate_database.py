from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm


relation_mappings = {
    'write_author_paper': 'author_paper_author',
}


@dataclass(frozen=True)
class Relation:
    subject: str
    relation: str
    object: Optional[str]


def parse_relation(rel_tuple: List) -> Relation:
    if len(rel_tuple) == 2:
        relation_string, subject = rel_tuple
        object = None
    else:
        relation_string, subject, object = rel_tuple
    # if relation_string == 'write_author_paper':
    #     relation = Relation(object, 'author_paper_author', subject)
    # else:
    relation = Relation(subject, relation_string, object)

    return relation


def bootstrap_database(source_db_path: Path, output_db_path: Path):
    assert source_db_path != output_db_path

    source_triple_store = {}

    source_relations = [
        parse_relation(line.strip().split('\t'))
        for line
        in source_db_path.open()
    ]

    # add the reverse of `write_author_paper` -> `author_paper_author`
    for relation in source_relations:
        if relation.relation == 'write_author_paper':
            new_relation = Relation(relation.object, 'author_paper_author', relation.subject)
            source_relations.append(new_relation)

    for relation in source_relations:
        source_triple_store.setdefault(relation.subject, dict()).setdefault(relation.relation, []).append(relation.object)

    relations = []
    for relation in tqdm(source_relations, desc='Bootstrapping...'):
        relations.append(relation)

        if relation.relation == 'cites_paper_paper':
            # 'cites_paper_author'
            obj_authors = source_triple_store.get(relation.object, {}).get('author_paper_author', [])
            for author in obj_authors:
                rel = Relation(relation.subject, 'cites_paper_author', author)
                relations.append(rel)

            # 'cites_author_paper'
            sub_authors = source_triple_store.get(relation.subject, {}).get('author_paper_author', [])
            for author in sub_authors:
                rel = Relation(author, 'cites_author_paper', relation.object)
                relations.append(rel)

            # 'cites_author_author'
            for sub_auhtor in sub_authors:
                for obj_author in obj_authors:
                    rel = Relation(sub_auhtor, 'cites_author_author', obj_author)
                    relations.append(rel)
        elif relation.relation == 'keyphrase_paper_keyphrase':
            # 'keyphrase_author_keyphrase'
            sub_authors = source_triple_store[relation.subject].get('author_paper_author', [])
            for author in sub_authors:
                rel = Relation(author, 'keyphrase_author_keyphrase', relation.object)
                relations.append(rel)

            # 'keyphrase_venue_keyphrase'
            for venue in source_triple_store[relation.subject].get('venue_paper_venue', []):
                rel = Relation(venue, 'keyphrase_venue_keyphrase', relation.object)
                relations.append(rel)

            # 'keyphrase_journal_keyphrase'
            for journal in source_triple_store[relation.subject].get('journal_paper_journal', []):
                rel = Relation(journal, 'keyphrase_journal_keyphrase', relation.object)
                relations.append(rel)
        elif relation.relation == 'author_paper_author':
            # 'publishin_author_venue'
            for venue in source_triple_store[relation.subject].get('venue_paper_venue', []):
                rel = Relation(relation.object, 'publishin_author_venue', venue)
                relations.append(rel)

            # 'publishin_author_journal'
            for journal in source_triple_store[relation.subject].get('journal_paper_journal', []):
                rel = Relation(relation.object, 'publishin_author_journal', journal)
                relations.append(rel)

    print(f'Num. total bootstrapped relations: {len(relations)}')
    unique_relations = list(set(relations))
    print(f'Num. unique bootstrapped relations: {len(unique_relations)}')
    unique_relations.sort(key=lambda relation: (relation.relation, relation.subject, relation.object))

    with output_db_path.open('w') as f:
        for relation in unique_relations:
            f.write(f'{relation.relation}\t{relation.subject}\t{relation.object}\n')


def bootstrap_geo_database(source_db_path: Path, output_db_path: Path):
    assert source_db_path != output_db_path

    source_triple_store = {}

    source_relations = [
        parse_relation(line.strip().split('\t'))
        for line
        in source_db_path.open()
    ]

    for relation in source_relations:
        source_triple_store.setdefault(
            relation.subject, dict()
        ).setdefault(relation.relation, []).append(relation.object)

    relations = []
    for relation in tqdm(source_relations, desc='Bootstrapping...'):
        relations.append(relation)

        if relation.relation == 'traverse_river_state':
            # 'loc_river_country'
            obj_countries = source_triple_store.get(relation.object, {}).get('loc_state_country', [])
            for country in obj_countries:
                rel = Relation(relation.subject, 'traverse_river_country', country)
                relations.append(rel)

        elif relation.relation == 'loc_mountain_state':
            # 'loc_mountain_country'
            obj_countries = source_triple_store.get(relation.object, {}).get('loc_state_country', [])
            for country in obj_countries:
                rel = Relation(relation.subject, 'loc_mountain_country', country)
                relations.append(rel)

        elif relation.relation == 'loc_place_state':
            # 'loc_place_country'
            obj_countries = source_triple_store.get(relation.object, {}).get('loc_state_country', [])
            for country in obj_countries:
                rel = Relation(relation.subject, 'loc_place_country', country)
                relations.append(rel)

        elif relation.relation == 'loc_city_state':
            # 'loc_city_country'
            obj_countries = source_triple_store.get(relation.object, {}).get('loc_state_country', [])
            for country in obj_countries:
                rel = Relation(relation.subject, 'loc_city_country', country)
                relations.append(rel)

    print(f'Num. total bootstrapped relations: {len(relations)}')
    unique_relations = list(set(relations))
    top_dup_rels = [key for key, val in Counter(relations).items() if val > 1][:10]
    print('Some duplicated relations')
    for rel in top_dup_rels:
        print(rel)
    print(f'Num. unique bootstrapped relations: {len(unique_relations)}')
    unique_relations.sort(key=lambda relation: (relation.relation, relation.subject, relation.object if relation.object else ''))

    with output_db_path.open('w') as f:
        for relation in unique_relations:
            if relation.object:
                f.write(f'{relation.relation}\t{relation.subject}\t{relation.object}\n')
            else:
                f.write(f'{relation.relation}\t{relation.subject}\n')

    old_db_lines = [l.strip() for l in source_db_path.open()]

    top_dup_rels = [key for key, val in Counter(old_db_lines).items() if val > 1][:10]
    print('Some duplicated relations in OLD database')
    for rel in top_dup_rels:
        print(rel)

    new_db_lines = set(l.strip() for l in output_db_path.open())

    assert all(
        line in new_db_lines for line in old_db_lines
    )


if __name__ == '__main__':
    # bootstrap_database(
    #     Path('lib/data/overnight/dbs/scholar.db'),
    #     Path('lib/data/overnight/dbs/scholar.bootstrapped.db'),
    # )

    bootstrap_geo_database(
        Path('lib/data/overnight/geo880.db'),
        Path('lib/data/overnight/geo880.bootstrapped.db'),
    )
