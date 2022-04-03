from typing import Dict, List

import torch
from paraphrase.sim.sim_models import WordAveraging
from paraphrase.sim.sim_utils import Example
from nltk.tokenize import TreebankWordTokenizer
import sentencepiece as spm


def make_example(sentence, tok_model, sp_model, model):
    sentence = sentence.lower()
    sentence = " ".join(tok_model.tokenize(sentence))
    sentence = sp_model.EncodeAsPieces(sentence)
    wp1 = Example(" ".join(sentence))
    wp1.populate_embeddings(model.vocab)
    return wp1


def encode_sentences(sentences: List[str], model: Dict) -> torch.Tensor:
    examples = [
        make_example(
            sentence,
            model['tokenizer'],
            model['spm'],
            model['model']
        )
        for sentence
        in sentences
    ]

    idxs, lengths, masks = model['model'].torchify_batch(examples)
    sent_vecs = model['model'].encode(idxs, masks, lengths)

    return sent_vecs


def find_similarity(s1, s2, tok_model, sp_model, model):
    with torch.no_grad():
        s1 = [make_example(x, tok_model, sp_model, model) for x in s1]
        s2 = [make_example(x, tok_model, sp_model, model) for x in s2]
        wx1, wl1, wm1 = model.torchify_batch(s1)
        wx2, wl2, wm2 = model.torchify_batch(s2)
        scores = model.scoring_function(wx1, wm1, wl1, wx2, wm2, wl2)
        return [x.item() for x in scores]


def main():
    tok = TreebankWordTokenizer()

    model = torch.load('paraphrase/sim/sim.pt', 'cpu')
    state_dict = model['state_dict']
    vocab_words = model['vocab_words']
    args = model['args']
    args.gpu = -1
    # turn off gpu
    model = WordAveraging(args, vocab_words)
    model.load_state_dict(state_dict, strict=True)
    sp = spm.SentencePieceProcessor()
    sp.Load('paraphrase/sim/sim.sp.30k.model')
    model.eval()

    s1 = "most cited author in deep learning"
    s2 = "who is the most popular author in deep learning?"
    print(find_similarity([s1], [s2], tok, sp, model))


if __name__ == '__main__':
    main()
