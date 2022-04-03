import string
from typing import Dict

from grammar_generation.generate_dataset_from_sampled_examples import PLACEHOLDER_ENTITY_MAP


class UtteranceRealizer:
    def __init__(self):
        self.fb_entity_to_info = {}
        for entity_name, entry in PLACEHOLDER_ENTITY_MAP.items():
            if entry['type_name'] not in {'year', 'misc'}:
                self.fb_entity_to_info[entry['entity']] = {
                    'name': entity_name,
                    'entry': entry
                }
            else:
                self.fb_entity_to_info['$'.join(entry['entity'])] = {
                    'name': entity_name,
                    'entry': entry
                }

    def realize(self, utterance: str, variable_dict: Dict) -> str:
        for slot_id, entity_id in variable_dict.items():
            if isinstance(entity_id, list):
                entity_id = entity_id[1]

            if entity_id in self.fb_entity_to_info:
                entity_name = self.fb_entity_to_info[entity_id]['name']
            else:
                assert isinstance(entity_id, (int, str, float))

                entity_name = entity_id

            utterance = utterance.replace(
                slot_id,
                entity_name
            )

        return utterance


class FixedEntityUtteranceRealizer:
    def __init__(self, domain='scholar'):
        from grammar_generation.generate_dataset_from_sampled_examples import PLACEHOLDER_ENTITY_MAP
        named_entities = sorted(PLACEHOLDER_ENTITY_MAP)
        typed_slot_to_entity_name_map = {}

        self.entity_vocab = set(PLACEHOLDER_ENTITY_MAP)

        for named_entity in named_entities:
            entry = PLACEHOLDER_ENTITY_MAP[named_entity]
            typed_slot_name = f'{entry["type_name"]}0'
            alternative_typed_slot_name = f'{entry["type_name"]}1'

            if typed_slot_name not in typed_slot_to_entity_name_map:
                typed_slot_to_entity_name_map[typed_slot_name] = named_entity
            elif alternative_typed_slot_name not in typed_slot_to_entity_name_map:
                typed_slot_to_entity_name_map[alternative_typed_slot_name] = named_entity

        self.typed_slot_to_entity_name_map = typed_slot_to_entity_name_map

        print(typed_slot_to_entity_name_map)

    def realize(self, utterance: str, variables: Dict, un_capitalize=False, normalize_end_punct=False):
        for slot_name in variables:
            utterance = utterance.replace(
                slot_name,
                self.typed_slot_to_entity_name_map[slot_name]
            )

        if un_capitalize and not(
            any(
                utterance.lower().startswith(entity.lower())
                for entity
                in self.entity_vocab
            )
        ):
            utterance = utterance[0].lower() + utterance[1:]

        if normalize_end_punct:
            if utterance[-1] in string.punctuation:
                utterance = utterance[:-1].strip()

        return utterance


if __name__ == '__main__':
    realizer = FixedEntityUtteranceRealizer()
    print(realizer.realize('ICML have Dan Klein and Tom Mitchell written ?', ['title0'], un_capitalize=True, normalize_end_punct=True))