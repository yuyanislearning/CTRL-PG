# coding=utf-8
# Copyright 2019 The Open AI Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for OpenAI GPT."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json
import logging
import os
import re
import unicodedata
from io import open

import sacremoses as sm

from .tokenization_utils import PreTrainedTokenizer
from .tokenization_bert import BasicTokenizer

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
    'vocab_file': 'vocab.json',
    'merges_file': 'merges.txt',
}

PRETRAINED_VOCAB_FILES_MAP = {
    'vocab_file':
    {
        'xlm-mlm-en-2048': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-en-2048-vocab.json",
        'xlm-mlm-ende-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-ende-1024-vocab.json",
        'xlm-mlm-enfr-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enfr-1024-vocab.json",
        'xlm-mlm-enro-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enro-1024-vocab.json",
        'xlm-mlm-tlm-xnli15-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-tlm-xnli15-1024-vocab.json",
        'xlm-mlm-xnli15-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-xnli15-1024-vocab.json",
        'xlm-clm-enfr-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-clm-enfr-1024-vocab.json",
        'xlm-clm-ende-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-clm-ende-1024-vocab.json",
    },
    'merges_file':
    {
        'xlm-mlm-en-2048': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-en-2048-merges.txt",
        'xlm-mlm-ende-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-ende-1024-merges.txt",
        'xlm-mlm-enfr-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enfr-1024-merges.txt",
        'xlm-mlm-enro-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enro-1024-merges.txt",
        'xlm-mlm-tlm-xnli15-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-tlm-xnli15-1024-merges.txt",
        'xlm-mlm-xnli15-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-xnli15-1024-merges.txt",
        'xlm-clm-enfr-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enfr-1024-merges.txt",
        'xlm-clm-ende-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-ende-1024-merges.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'xlm-mlm-en-2048': 512,
    'xlm-mlm-ende-1024': 512,
    'xlm-mlm-enfr-1024': 512,
    'xlm-mlm-enro-1024': 512,
    'xlm-mlm-tlm-xnli15-1024': 512,
    'xlm-mlm-xnli15-1024': 512,
    'xlm-clm-enfr-1024': 512,
    'xlm-clm-ende-1024': 512,
}

def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('´', "'")
    text = re.sub(r'''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
    text = re.sub(r'\s*\n\s*', ' \n ', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text.strip()


def lowercase_and_remove_accent(text):
    """
    Lowercase and strips accents from a piece of text based on
    https://github.com/facebookresearch/XLM/blob/master/tools/lowercase_and_remove_accent.py
    """
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output).lower()


def replace_unicode_punct(text):
    '''
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/replace-unicode-punctuation.perl
    '''
    text = text.replace('，', ',')
    text = text.replace('。 *', '. ')
    text = text.replace('、', ',')
    text = text.replace('”', '"')
    text = text.replace('“', '"')
    text = text.replace('∶', ':')
    text = text.replace('：', ':')
    text = text.replace('？', '?')
    text = text.replace('《', '"')
    text = text.replace('》', '"')
    text = text.replace('）', ')')
    text = text.replace('！', '!')
    text = text.replace('（', '(')
    text = text.replace('；', ';')
    text = text.replace('１', '"')
    text = text.replace('」', '"')
    text = text.replace('「', '"')
    text = text.replace('０', '0')
    text = text.replace('３', '3')
    text = text.replace('２', '2')
    text = text.replace('５', '5')
    text = text.replace('６', '6')
    text = text.replace('９', '9')
    text = text.replace('７', '7')
    text = text.replace('８', '8')
    text = text.replace('４', '4')
    text = re.sub(r'．\s*', '. ', text)
    text = text.replace('～', '~')
    text = text.replace('’', '\'')
    text = text.replace('…', '...')
    text = text.replace('━', '-')
    text = text.replace('〈', '<')
    text = text.replace('〉', '>')
    text = text.replace('【', '[')
    text = text.replace('】', ']')
    text = text.replace('％', '%')
    return text


def remove_non_printing_char(text):
    '''
    Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/remove-non-printing-char.perl
    '''
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat.startswith('C'):
            continue
        output.append(char)
    return "".join(output)


def romanian_preprocessing(text):
    '''Sennrich's WMT16 scripts for Romanian preprocessing, used by model `xlm-mlm-enro-1024`'''
    # https://github.com/rsennrich/wmt16-scripts/blob/master/preprocess/normalise-romanian.py
    text = text.replace("\u015e", "\u0218").replace("\u015f", "\u0219")
    text = text.replace("\u0162", "\u021a").replace("\u0163", "\u021b")
    # https://github.com/rsennrich/wmt16-scripts/blob/master/preprocess/remove-diacritics.py
    text = text.replace("\u0218", "S").replace("\u0219", "s") #s-comma
    text = text.replace("\u021a", "T").replace("\u021b", "t") #t-comma
    text = text.replace("\u0102", "A").replace("\u0103", "a")
    text = text.replace("\u00C2", "A").replace("\u00E2", "a")
    text = text.replace("\u00CE", "I").replace("\u00EE", "i")
    return text


class XLMTokenizer(PreTrainedTokenizer):
    """
    BPE tokenizer for XLM, adapted from OpenAI BPE tokenizer. Peculiarities:

        - lower case all inputs

        - uses `SpaCy tokenizer <https://spacy.io/api/tokenizer/>`_ and \
        `ftfy <https://ftfy.readthedocs.io/en/latest/>`_ for pre-BPE tokenization if they are installed, \
        fallback to BERT's BasicTokenizer if not.

        - argument ``special_tokens`` and function ``set_special_tokens``, can be used to add additional symbols \
        (ex: "__classify__") to a vocabulary.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, merges_file, unk_token="<unk>", bos_token="<s>",
                 sep_token="</s>", pad_token="<pad>", cls_token="</s>",
                 mask_token="<special1>", additional_special_tokens=["<special0>",
                 "<special1>", "<special2>", "<special3>", "<special4>", "<special5>",
                 "<special6>", "<special7>", "<special8>", "<special9>"], **kwargs):
        super(XLMTokenizer, self).__init__(unk_token=unk_token, bos_token=bos_token,
                                           sep_token=sep_token, pad_token=pad_token,
                                           cls_token=cls_token, mask_token=mask_token,
                                           additional_special_tokens=additional_special_tokens,
                                           **kwargs)

        # cache of sm.MosesPunctNormalizer instance
        self.cache_moses_punct_normalizer = dict()
        # cache of sm.MosesTokenizer instance
        self.cache_moses_tokenizer = dict()
        self.lang_with_custom_tokenizer = set(['zh', 'th', 'ja'])
        # True for current supported model (v1.2.0), False for XLM-17 & 100
        self.do_lowercase_and_remove_accent = True

        self.encoder = json.load(open(vocab_file, encoding="utf-8"))
        self.decoder = {v:k for k,v in self.encoder.items()}
        merges = open(merges_file, encoding='utf-8').read().split('\n')[:-1]
        merges = [tuple(merge.split()[:2]) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    def moses_punct_norm(self, text, lang):
        if lang not in self.cache_moses_punct_normalizer:
            punct_normalizer = sm.MosesPunctNormalizer(lang=lang)
            self.cache_moses_punct_normalizer[lang] = punct_normalizer
        else:
            punct_normalizer = self.cache_moses_punct_normalizer[lang]
        return punct_normalizer.normalize(text)

    def moses_tokenize(self, text, lang):
        if lang not in self.cache_moses_tokenizer:
            moses_tokenizer = sm.MosesTokenizer(lang=lang)
            self.cache_moses_tokenizer[lang] = moses_tokenizer
        else:
            moses_tokenizer = self.cache_moses_tokenizer[lang]
        return moses_tokenizer.tokenize(text, return_str=False, escape=False)

    def moses_pipeline(self, text, lang):
        text = replace_unicode_punct(text)
        text = self.moses_punct_norm(text, lang)
        text = remove_non_printing_char(text)
        return text

    @property
    def vocab_size(self):
        return len(self.encoder)

    def bpe(self, token):
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        if word == '\n  </w>':
            word = '\n</w>'
        self.cache[token] = word
        return word

    def _tokenize(self, text, lang='en'):
        """ Tokenize a string. """
        split_tokens = []
        if self.do_lowercase_and_remove_accent:
            text = lowercase_and_remove_accent(text)
        if lang not in self.lang_with_custom_tokenizer:
            text = self.moses_pipeline(text, lang=lang)
            # TODO: make sure we are using `xlm-mlm-enro-1024`, since XLM-100 doesn't have this step
            if lang == 'ro':
                text = romanian_preprocessing(text)
            text = self.moses_tokenize(text, lang=lang)
            for token in text:
                split_tokens.extend([t for t in self.bpe(token).split(' ')])
        else:
            raise ValueError
        return split_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = ''.join(tokens).replace('</w>', ' ').strip()
        return out_string

    def add_special_tokens_single_sentence(self, token_ids):
        """
        Adds special tokens to a sequence for sequence classification tasks.
        An XLM sequence has the following format: [CLS] X [SEP]
        """
        return [self._convert_token_to_id(self.cls_token)] + token_ids + [self._convert_token_to_id(self.sep_token)]

    def add_special_tokens_sentences_pair(self, token_ids_0, token_ids_1):
        """
        Adds special tokens to a sequence pair for sequence classification tasks.
        An XLM sequence pair has the following format: [CLS] A [SEP] B [SEP]
        """
        sep = [self._convert_token_to_id(self.sep_token)]
        cls = [self._convert_token_to_id(self.cls_token)]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def save_vocabulary(self, save_directory):
        """Save the tokenizer vocabulary and merge files to a directory."""
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        vocab_file = os.path.join(save_directory, VOCAB_FILES_NAMES['vocab_file'])
        merge_file = os.path.join(save_directory, VOCAB_FILES_NAMES['merges_file'])

        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.encoder, ensure_ascii=False))

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning("Saving vocabulary to {}: BPE merge indices are not consecutive."
                                   " Please check that the tokenizer is not corrupted!".format(merge_file))
                    index = token_index
                writer.write(' '.join(bpe_tokens) + u'\n')
                index += 1

        return vocab_file, merge_file
