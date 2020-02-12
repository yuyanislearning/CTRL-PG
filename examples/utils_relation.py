# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """

from __future__ import absolute_import, division, print_function

import logging
import os
from io import open
import csv
import sys
import copy
import json
import numpy as np
import random

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class Input_Graph_Example(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    def __init__(self, guid, text, matrix, relations, doc_id):
        self.guid = guid
        self.text = text
        self.matrix = matrix
        self.relations = relations
        self.doc_id = doc_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class Input_SB_Example(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    def __init__(self, guid, text,  relations, doc_id, sen_id):
        self.guid = guid
        self.text = text
        #self.matrix = matrix
        self.relations = relations
        self.doc_id = doc_id
        self.sen_id = sen_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class Input_Graph_Features(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_masks, token_type_ids,  relations, doc_id):#matrix,
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        #self.matrix = matrix
        self.relations = relations
        self.doc_id = doc_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class Input_SB_Features(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_masks, token_type_ids,  relations, doc_id, sen_id, ids, sources,node_pos):#matrix,
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        #self.matrix = matrix
        self.relations = relations
        self.doc_id = doc_id
        self.sen_id = sen_id
        self.ids = ids
        self.sources = sources
        self.node_pos = node_pos

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors

        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def tfds_map(self, example):
        """Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are. 
        This method converts examples to the correct format."""
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

#class InputExample(object):
#    """A single training/test example for token classification."""
#
#    def __init__(self, guid, words, labels):
#        """Constructs a InputExample.
#
#        Args:
#            guid: Unique id for the example.
#            words: list. The words of the sequence.
#            labels: (Optional) list. The labels for each word of the sequence. This should be
#            specified for train and dev examples, but not for test examples.
#        """
#        self.guid = guid
#        self.words = words
#        self.labels = labels


#class InputFeatures(object):
#    """A single set of features of data."""
#
#    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
#        self.input_ids = input_ids
#        self.input_mask = input_mask
#        self.segment_ids = segment_ids
#        self.label_ids = label_ids


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(guid="{}-{}".format(mode, guid_index),
                                                 words=words,
                                                 labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(guid="%s-%d".format(mode, guid_index),
                                         words=words,
                                         labels=labels))
    return examples

def is_tf_available():
    return False

def glue_convert_examples_to_features(examples, tokenizer,
                                      max_length=512,
                                      task=None,
                                      label_list=None,
                                      output_mode=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True
    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        try:
            if output_mode == "classification":
                label = label_map[example.label]
            elif output_mode == "regression":
                label = float(example.label)
            else:       
                raise KeyError(output_mode)
        except:
            print(example.text_a)
            continue

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label=label))

    if is_tf_available() and is_tf_dataset:
        def gen():
            for ex in features:
                yield  ({'input_ids': ex.input_ids,
                         'attention_mask': ex.attention_mask,
                         'token_type_ids': ex.token_type_ids},
                        ex.label)

        return tf.data.Dataset.from_generator(gen,
            ({'input_ids': tf.int32,
              'attention_mask': tf.int32,
              'token_type_ids': tf.int32},
             tf.int64),
            ({'input_ids': tf.TensorShape([None]),
              'attention_mask': tf.TensorShape([None]),
              'token_type_ids': tf.TensorShape([None])},
             tf.TensorShape([])))

    return features

def sb_convert_examples_to_features(examples, tokenizer,
                                      max_length=64,
                                      task=None,
                                      label_list=None,
                                      output_mode=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True,
                                      data_aug = 'rules',
                                      evaluate = False,
                                      ):#max_node_size=650
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True
    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label.lower(): i for i, label in enumerate(label_list)}    

    features = []
    #max_length_input = 0

    dict_IndenToID = {}
    
    wrong_count = 0
    sen_count = 0

    for (ex_index, example) in enumerate(examples):
        sen_count+=1
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)

        input_ids, token_type_ids, attention_masks = [], [], []

        # for one document
        #rel = build_sb_relations(rel = example.relations)

        IDToIndex, IndexToID = IDIndexDic(rel = example.relations)
        dict_IndenToID[str(example.doc_id)+example.sen_id] = IndexToID
        if evaluate: 
            data_aug = 'rules'
        if data_aug == "reverse":
            for rel in example.relations:
                texts = [] 
                for text in example.text:
                    texts.extend(text)
                texts = texts[0:rel[0]] + ['<e1>'] + texts[rel[0]:(rel[1]+1)] + ['</e1>'] + texts[(rel[1]+1):rel[3]] + ['<e2>'] + texts[rel[3]:(rel[4]+1)] + ['</e2>'] + texts[(rel[4]+1):len(texts)]
                #print("This is texts:",texts)
                #print("This is rel",rel)
                
                texts = ' '.join(texts)
                inputs = tokenizer.encode_plus(
                    texts,
                    add_special_tokens=True,
                    max_length=max_length,
                )

                input_id, token_type_id = inputs["input_ids"], inputs["token_type_ids"]


                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                attention_mask = [1 if mask_padding_with_zero else 0] * len(input_id)

                # Zero-pad up to the sequence length.
                padding_length = max_length - len(input_id)

                if pad_on_left:
                    input_id = ([pad_token] * padding_length) + input_id
                    attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                    token_type_id = ([pad_token_segment_id] * padding_length) + token_type_id
                else:
                    input_id = input_id + ([pad_token] * padding_length)
                    attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                    token_type_id = token_type_id + ([pad_token_segment_id] * padding_length)

                assert len(input_id) == max_length, "Error with input length {} vs {}".format(len(input_id), max_length)
                assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
                assert len(token_type_id) == max_length, "Error with input length {} vs {}".format(len(token_type_id), max_length)

                # print("texts:",texts)
                # print("label", rel[6])
                # print("label", reltonum(rel[6]))
                # print("input_ids", input_id)
                # print("attention_mask", attention_mask)
                # print("token_type_id", token_type_id)
                # print("")

                features.append(
                    Input_SB_Features(input_ids=input_id,
                                attention_masks=attention_mask,
                                token_type_ids=token_type_id,
                                #matrix=example.matrix,
                                relations=reltonum(rel[6]),
                                doc_id=example.doc_id,
                                sen_id=example.sen_id,
                                ids=(IDToIndex[rel[2]], IDToIndex[rel[5]])
                                ))           
                texts = [] 
                for text in example.text:
                    texts.extend(text)
                texts = texts[0:rel[0]] + ['<e2>'] + texts[rel[0]:(rel[1]+1)] + ['</e2>'] + texts[(rel[1]+1):rel[3]] + ['<e1>'] + texts[rel[3]:(rel[4]+1)] + ['</e1>'] + texts[(rel[4]+1):len(texts)]
                #print("This is texts:",texts)
                #print("This is rel",rel)
                
                texts = ' '.join(texts)
                inputs = tokenizer.encode_plus(
                    texts,
                    add_special_tokens=True,
                    max_length=max_length,
                )

                input_id, token_type_id = inputs["input_ids"], inputs["token_type_ids"]

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                attention_mask = [1 if mask_padding_with_zero else 0] * len(input_id)

                # Zero-pad up to the sequence length.
                padding_length = max_length - len(input_id)

                if pad_on_left:
                    input_id = ([pad_token] * padding_length) + input_id
                    attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                    token_type_id = ([pad_token_segment_id] * padding_length) + token_type_id
                else:
                    input_id = input_id + ([pad_token] * padding_length)
                    attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                    token_type_id = token_type_id + ([pad_token_segment_id] * padding_length)

                assert len(input_id) == max_length, "Error with input length {} vs {}".format(len(input_id), max_length)
                assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
                assert len(token_type_id) == max_length, "Error with input length {} vs {}".format(len(token_type_id), max_length)

                features.append(
                    Input_SB_Features(input_ids=input_id,
                                attention_masks=attention_mask,
                                token_type_ids=token_type_id,
                                #matrix=example.matrix,
                                relations=reverse(reltonum(rel[6])),
                                doc_id=example.doc_id,
                                sen_id=example.sen_id,
                                ids=(IDToIndex[rel[5]], IDToIndex[rel[2]])
                                )) 
 
        elif data_aug == "rules":
            BM, OM, IDM, pos_dict = build_BO(rel = example.relations, IDToIndex= IDToIndex)
            #print("origin BM", BM)
            BM, OM, wrong_count = iter_rule_update(BM, OM,  1,wrong_count)
            print("wrong count ",wrong_count)
            print("number of sentence", sen_count)
            #print("update BM", BM)
            AM = BM.transpose()
            temp_BM = (BM - IDM)*2
            temp_OM = (OM - IDM)*2
            temp_AM = (AM - IDM)*2
            temp_BM[np.where(temp_BM<0)] = 0
            temp_OM[np.where(temp_OM<0)] = 0
            temp_AM[np.where(temp_AM<0)] = 0
            IDM = IDM + temp_BM + temp_OM + temp_AM# TODO modify case 4



            texts = []
            for text in example.text:
                texts.extend(text)
            
            add_features(features, BM, pos_dict,IDM, 'BM', tokenizer , texts, example.doc_id,example.sen_id, max_length,mask_padding_with_zero,pad_token,pad_token_segment_id, pad_on_left)
 
            AM = BM.transpose()
            add_features(features, AM, pos_dict,IDM,'AM', tokenizer , texts, example.doc_id, example.sen_id, max_length,mask_padding_with_zero,pad_token,pad_token_segment_id,pad_on_left)

            add_features(features, OM, pos_dict,IDM, 'OM', tokenizer , texts, example.doc_id, example.sen_id, max_length,mask_padding_with_zero,pad_token,pad_token_segment_id,pad_on_left)

        else:
            for rel in example.relations:
                texts = [] 
                for text in example.text:
                    texts.extend(text)
                texts = texts[0:rel[0]] + ['<e1>'] + texts[rel[0]:(rel[1]+1)] + ['</e1>'] + texts[(rel[1]+1):rel[3]] + ['<e2>'] + texts[rel[3]:(rel[4]+1)] + ['</e2>'] + texts[(rel[4]+1):len(texts)]
                #print("This is texts:",texts)
                #print("This is rel",rel)
                
                texts = ' '.join(texts)
                inputs = tokenizer.encode_plus(
                    texts,
                    add_special_tokens=True,
                    max_length=max_length,
                )

                input_id, token_type_id = inputs["input_ids"], inputs["token_type_ids"]


                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                attention_mask = [1 if mask_padding_with_zero else 0] * len(input_id)

                # Zero-pad up to the sequence length.
                padding_length = max_length - len(input_id)

                if pad_on_left:
                    input_id = ([pad_token] * padding_length) + input_id
                    attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                    token_type_id = ([pad_token_segment_id] * padding_length) + token_type_id
                else:
                    input_id = input_id + ([pad_token] * padding_length)
                    attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                    token_type_id = token_type_id + ([pad_token_segment_id] * padding_length)

                assert len(input_id) == max_length, "Error with input length {} vs {}".format(len(input_id), max_length)
                assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
                assert len(token_type_id) == max_length, "Error with input length {} vs {}".format(len(token_type_id), max_length)

                features.append(
                    Input_SB_Features(input_ids=input_id,
                                attention_masks=attention_mask,
                                token_type_ids=token_type_id,
                                #matrix=example.matrix,
                                relations=reltonum(rel[6]),
                                doc_id=example.doc_id,
                                sen_id=example.sen_id,
                                ids=(IDToIndex[rel[2]], IDToIndex[rel[5]])
                                ))

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     #logger.info("node number: %s" % str(len(input_id)))
        #     logger.info("node dim: %s " % str(np.shape(input_id)))

        # if len(input_ids) > max_length_input: max_length_input = len(input_ids)
        # logger.info("current length: %s" % (str(len(input_ids))))
        # logger.info("max length: %s" % (str(max_length_input)))
        '''
        max_node_size = 10000
        padding_size = max_node_size - len(input_ids)
        if padding_size <0:
            input_ids = input_ids[0:(max_node_size-1)]
            attention_masks = attention_masks[0:(max_node_size-1)]
            token_type_ids = token_type_ids[0:(max_node_size-1)]
        else:
            input_ids = input_ids + ([[pad_token] * max_length] * padding_size)
            attention_masks = attention_masks + ([[0 if mask_padding_with_zero else 1] * max_length] * padding_size)
            token_type_ids = token_type_ids + ([[pad_token_segment_id] * max_length] * padding_size)
        '''
            #logger.info("text: %s" % (text))
            #logger.info("input_ids: %s" % " ".join([str(x) for x in input_id]))
            #logger.info("relations: %s" % " ".join([str(x) for x in relations[:5]]))
            #logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            #logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
    print("wrong number", wrong_count)
    #exit()
    return features, dict_IndenToID


def graph_convert_examples_to_features2(examples, tokenizer,
                                      max_length=64,
                                      task=None,
                                      label_list=None,
                                      output_mode=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True,
                                      max_node_size=650):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True
    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label.lower(): i for i, label in enumerate(label_list)}    

    features = []
    #max_length_input = 0

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)

        input_ids, token_type_ids, attention_masks = [], [], []

        # for one document
        emb  = build_PSL_dataset(adj = example.matrix, rel = example.relations,no_rule = True, random_sampling = False, n_rule = 100)
        texts = [text for text in example.text]
        for i, j,r in emb:
            text1 = texts[i].replace('<e>','<e1>')
            text1 = texts[i].replace('</e>','</e1>')
            text2 = texts[j].replace('<e>','<e2>')
            text2 = texts[j].replace('</e>','</e2>')
            text = text1 + text2
            inputs = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
            )

            input_id, token_type_id = inputs["input_ids"], inputs["token_type_ids"]


            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_id)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_id)

            if pad_on_left:
                input_id = ([pad_token] * padding_length) + input_id
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_id = ([pad_token_segment_id] * padding_length) + token_type_id
            else:
                input_id = input_id + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_id = token_type_id + ([pad_token_segment_id] * padding_length)

            assert len(input_id) == max_length, "Error with input length {} vs {}".format(len(input_id), max_length)
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
            assert len(token_type_id) == max_length, "Error with input length {} vs {}".format(len(token_type_id), max_length)

            
            features.append(
                Input_Graph_Features(input_ids=input_id,
                            attention_masks=attention_mask,
                            token_type_ids=token_type_id,
                            #matrix=example.matrix,
                            relations=r,
                            doc_id=example.doc_id
                            ))
            '''
            # augment data by reverse e1,e2 and r
            text1 = texts[i].replace('<e>','<e2>')
            text2 = texts[j].replace('<e>','<e1>')
            text1 = texts[i].replace('</e>','</e2>')
            text2 = texts[j].replace('</e>','</e1>')
            text = text1 + text2
            inputs = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
            )

            input_id, token_type_id = inputs["input_ids"], inputs["token_type_ids"]


            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_id)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_id)

            if pad_on_left:
                input_id = ([pad_token] * padding_length) + input_id
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_id = ([pad_token_segment_id] * padding_length) + token_type_id
            else:
                input_id = input_id + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_id = token_type_id + ([pad_token_segment_id] * padding_length)

            assert len(input_id) == max_length, "Error with input length {} vs {}".format(len(input_id), max_length)
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
            assert len(token_type_id) == max_length, "Error with input length {} vs {}".format(len(token_type_id), max_length)

            
            features.append(
                Input_Graph_Features(input_ids=input_id,
                            attention_masks=attention_mask,
                            token_type_ids=token_type_id,
                            #matrix=example.matrix,
                            relations=reverse(r),
                            doc_id=example.doc_id
                            ))
            '''
        #relations = emb
        #print("emb",emb)

        # if len(input_ids) > max_length_input: max_length_input = len(input_ids)
        # logger.info("current length: %s" % (str(len(input_ids))))
        # logger.info("max length: %s" % (str(max_length_input)))
        '''
        max_node_size = 10000
        padding_size = max_node_size - len(input_ids)
        if padding_size <0:
            input_ids = input_ids[0:(max_node_size-1)]
            attention_masks = attention_masks[0:(max_node_size-1)]
            token_type_ids = token_type_ids[0:(max_node_size-1)]
        else:
            input_ids = input_ids + ([[pad_token] * max_length] * padding_size)
            attention_masks = attention_masks + ([[0 if mask_padding_with_zero else 1] * max_length] * padding_size)
            token_type_ids = token_type_ids + ([[pad_token_segment_id] * max_length] * padding_size)
        '''
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("node number: %s" % str(len(input_ids)))
            logger.info("node dim: %s " % str(np.shape(input_ids)))
            #logger.info("text: %s" % (text))
            #logger.info("input_ids: %s" % " ".join([str(x) for x in input_id]))
            #logger.info("relations: %s" % " ".join([str(x) for x in relations[:5]]))
            #logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            #logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))




    # if is_tf_available() and is_tf_dataset:
    #     def gen():
    #         for ex in features:
    #             yield  ({'input_ids': ex.input_ids,
    #                      'attention_mask': ex.attention_mask,
    #                      'token_type_ids': ex.token_type_ids},
    #                     ex.label)

    #     return tf.data.Dataset.from_generator(gen,
    #         ({'input_ids': tf.int32,
    #           'attention_mask': tf.int32,
    #           'token_type_ids': tf.int32},
    #          tf.int64),
    #         ({'input_ids': tf.TensorShape([None]),
    #           'attention_mask': tf.TensorShape([None]),
    #           'token_type_ids': tf.TensorShape([None])},
    #          tf.TensorShape([])))

    return features

def graph_convert_examples_to_features(examples, tokenizer,
                                      max_length=64,
                                      task=None,
                                      label_list=None,
                                      output_mode=None,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True,
                                      max_node_size=650):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True
    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label.lower(): i for i, label in enumerate(label_list)}    

    features = []
    #max_length_input = 0
    print("examples", len(examples))
    
    for (ex_index, example) in enumerate(examples):

        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)

        input_ids, token_type_ids, attention_masks = [], [], []
        emb  = build_eval_dataset( rel = example.relations)
        texts = [text for text in example.text]
        print("emb", len(emb))
        for i, j,r in emb:
            text1 = texts[i].replace('<e>','<e1>')
            text2 = texts[j].replace('<e>','<e2>')
            text1 = texts[i].replace('</e>','</e1>')
            text2 = texts[j].replace('</e>','</e2>')
            text = text1 + text2
            inputs = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
            )

            input_id, token_type_id = inputs["input_ids"], inputs["token_type_ids"]


            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_id)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_id)

            if pad_on_left:
                input_id = ([pad_token] * padding_length) + input_id
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_id = ([pad_token_segment_id] * padding_length) + token_type_id
            else:
                input_id = input_id + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_id = token_type_id + ([pad_token_segment_id] * padding_length)

            assert len(input_id) == max_length, "Error with input length {} vs {}".format(len(input_id), max_length)
            assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
            assert len(token_type_id) == max_length, "Error with input length {} vs {}".format(len(token_type_id), max_length)
            features.append(
                    Input_Graph_Features(input_ids=input_id,
                                attention_masks=attention_mask,
                                token_type_ids=token_type_id,
                                #matrix=example.matrix,
                                relations=r,
                                doc_id=example.doc_id
                                ))
            '''
            input_ids.append(input_id)
            token_type_ids.append(token_type_id)
            attention_masks.append(attention_mask)'''

        #relations = emb

        # if len(input_ids) > max_length_input: max_length_input = len(input_ids)
        # logger.info("current length: %s" % (str(len(input_ids))))
        # logger.info("max length: %s" % (str(max_length_input)))
        '''
        padding_size = max_node_size - len(input_ids)
        input_ids = input_ids + ([[pad_token] * max_length] * padding_size)
        attention_masks = attention_masks + ([[0 if mask_padding_with_zero else 1] * max_length] * padding_size)
        token_type_ids = token_type_ids + ([[pad_token_segment_id] * max_length] * padding_size)
        '''
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            #logger.info("node number: %s" % str(len(input_ids)))
            #logger.info("node dim: %s " % str(np.shape(input_ids)))
            logger.info("text: %s" % (text))
                #logger.info("input_ids: %s" % " ".join([str(x) for x in input_id]))
                #logger.info("relations: %s" % " ".join([str(x) for x in relations[:5]]))
                #logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                #logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))




    # if is_tf_available() and is_tf_dataset:
    #     def gen():
    #         for ex in features:
    #             yield  ({'input_ids': ex.input_ids,
    #                      'attention_mask': ex.attention_mask,
    #                      'token_type_ids': ex.token_type_ids},
    #                     ex.label)

    #     return tf.data.Dataset.from_generator(gen,
    #         ({'input_ids': tf.int32,
    #           'attention_mask': tf.int32,
    #           'token_type_ids': tf.int32},
    #          tf.int64),
    #         ({'input_ids': tf.TensorShape([None]),
    #           'attention_mask': tf.TensorShape([None]),
    #           'token_type_ids': tf.TensorShape([None])},
    #          tf.TensorShape([])))
    #print("length of features", len(features))
    return features


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['premise'].numpy().decode('utf-8'),
                            tensor_dict['hypothesis'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['question1'].numpy().decode('utf-8'),
                            tensor_dict['question2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['question'].numpy().decode('utf-8'),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class I2b2Processor(DataProcessor):
    """Processor for the i2b2-m data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["OVERLAP", "BEFORE", "AFTER"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            try:
                text_a = line[0]+' '+line[1]
            #text_b = line[1]
                label = line[2]
            except:
                print(line)
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

class I2b2_Graph_Processor(DataProcessor):
    """Processor for the i2b2-m data set (GLUE version)."""


    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_labels(self):
        """See base class."""
        return ["overlap", "before", "after"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (key, val) in lines.items():

            guid = "%s-%s" % (set_type, int(key))
            try:
                text = val['nodes']
                matrix = val['matrix']
                relations = val['relations']
                doc_id = int(key)
            except:
                print(line)
                continue
            examples.append(
                Input_Graph_Example(guid=guid, text=text, matrix=matrix, relations=relations, doc_id=doc_id))
        return examples

    def _read_json(self, path):
        """Reads a tab separated value file."""
        return json.load(open(path))

class I2b2_SB_Processor(DataProcessor):
    """Processor for the i2b2-m data set (GLUE version)."""


    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_labels(self):
        """See base class."""
        return ["overlap", "before", "after"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (key, val) in lines.items():

            guid = "%s-%s" % (set_type, int(key))
            for skey, sval in val.items():
                # try:
                text = sval['text']
                relations = sval['relation']
                if len(relations)==0:
                    continue
                doc_id = int(key)
                sen_id = skey
                # except:
                #     print(lines)
                #     print("*****************************")
                examples.append(
                    Input_SB_Example(guid=guid, text=text,  relations=relations, doc_id=doc_id, sen_id=sen_id))
        return examples

    def _read_json(self, path):
        """Reads a tab separated value file."""
        return json.load(open(path))


def build_PSL_dataset(adj = None, rel = None, no_rule = True ,random_sampling = False, n_rule = 100):
    emb =[]
    if no_rule:
        for [j,k,r] in rel:
            if r=="OVERLAP":
                emb.append([j,k,0])
            if r=="BEFORE":
                emb.append([j,k,1])
            if r=="OVERLAP":
                emb.append([j,k,2])
        return emb
    # convert origin matrix to before matrix and after matrix
    # augment data from transivity rules
    n = len(adj)
    BM = np.zeros((n,n))
    OM = np.zeros((n,n))
    #print('rel',rel)
    for [j,k,r] in rel:
        if r=="OVERLAP":
            OM[j,k]=1
        if r=="BEFORE":
            BM[j,k]=1
        if r=="OVERLAP":
            BM[k,j]=1
    #print("Before Before:", 2*len(np.where(BM>0)[0]))
    #print("Before Overlap:", len(np.where(OM>0)[0]))
    BM, OM = iter_rule_update(BM, OM, n_iter = 5)
    #dense_adj = BM+OM+BM.transpose()
    #dense_adj[np.where(dense_adj>0)] = 1
    #print("Updated Before:", 2*len(np.where(BM>0)[0]))
    #print("UPdated Overlap:", len(np.where(OM>0)[0]))
    #B_link = len(np.where(BM>0)[0])
    #O_link = len(np.where(OM>0)[0])
    #print("connectivity: ", (B_link+O_link)/n/(n-1))

    # construct all rules tensor
    BBB = rule_tensor(BM, BM)
    BOB = rule_tensor(BM, OM)
    OBB = rule_tensor(OM, BM)
    OOO = rule_tensor(OM, OM)
    All_rules = BBB+BOB+OBB+OOO
    all_x, all_y, all_z = np.where(All_rules>0)
    print("rules in origin data:", all_x.shape)
    

    for [j,k,r] in rel:
        # for no rules found
        if np.where(all_x==j)[0].shape[0]==0 or np.where(all_y==k)[0].shape[0]==0 or len(set(np.where(all_x==j)[0]).intersection(set(np.where(all_y==k)[0])))==0:
            continue
            '''
            #emb.append(torch.tensor([j,k]))
            emb.extend([torch.tensor([j,k]),torch.tensor([j,k]),torch.tensor([j,k])])
            emb.extend([torch.tensor([k,j]),torch.tensor([k,j]),torch.tensor([k,j])])
            if r ==0:
                #labels.append(r)
                labels.extend([r,r,r])
                labels.extend([r,r,r])
            if r ==1:
                #labels.append(r)
                labels.extend([r,r,r])
                labels.extend([2,2,2])
            if r ==2:
                #labels.append(r)
                labels.extend([r,r,r])
                labels.extend([1,1,1])
            continue'''
        if r==2:
            j,k = k,j
        # sample from rules
        rule_exist = set(np.where(all_x==j)[0]).intersection(set(np.where(all_y==k)[0]))
        common_neigh = all_z[random.sample(rule_exist, 1)[0]]
        # build PSL dataset, and their sysmetric version
        '''
        rules encoding:
        BBB:0, AAA:1, BOB:2, AOA:3, OBB:4, OAA:5, OOO:6, None:7
        '''
        if BBB[j,k,common_neigh]>0:
            #emb.append([j,k,1])
            #emb.append([k,j,2])
            emb.extend([[j,k,1],[k,common_neigh,1]])
            emb.extend([[k,j,2],[common_neigh,k,2]])
            #labels.append(1)
        if BOB[j,k,common_neigh]>0:
            #emb.append([j,k, 1])
            #emb.append([k,j,2])
            emb.extend([[j,k,1],[k,common_neigh,0]])
            emb.extend([[k,j,2],[common_neigh,k,0]])
        if OBB[j,k,common_neigh]>0:
            #emb.append([j,k,0])
            #emb.append([k,j,0])
            emb.extend([[j,k,0],[k,common_neigh,1]])
            emb.extend([[k,j,0],[common_neigh,k,2]])
        if OOO[j,k,common_neigh]>0:
            #emb.append([j,k, 0])
            #emb.append([k,j,0])
            emb.extend([[j,k,0],[k,common_neigh,0]])
            emb.extend([[k,j,0],[common_neigh,k,0]])

    return emb

def add_features(features,M, pos_dict,IDM, M_type, tokenizer , texts,doc_id ,sen_id, max_length,mask_padding_with_zero,pad_token,pad_token_segment_id,pad_on_left):
    if M_type == "BM":
        r = 1
    if M_type == 'OM':
        r = 0
    if M_type == "AM":
        r = 2
    all_x, all_y = np.where(M>0)
    for i in range(all_x.size):
        x1, x2 = pos_dict[all_x[i]]
        y1, y2 = pos_dict[all_y[i]]
        # in case x>y
        if x1 > y1:
            x1,x2,y1,y2 = y1,y2,x1,x2
            new_text = texts[0:x1] + ['<e2>'] + texts[x1:(x2+1)] + ['</e2>'] + texts[(x2+1):y1] + ['<e1>'] + texts[y1:(y2+1)] + ['</e1>'] + texts[(y2+1):len(texts)]
            new_text = ' '.join(new_text)
        else:
            new_text = texts[0:x1] + ['<e1>'] + texts[x1:(x2+1)] + ['</e1>'] + texts[(x2+1):y1] + ['<e2>'] + texts[y1:(y2+1)] + ['</e2>'] + texts[(y2+1):len(texts)]
            new_text = ' '.join(new_text)

        inputs = tokenizer.encode_plus(
            new_text,
            add_special_tokens=True,
            max_length=max_length,
        )
        node_pos_1 = node_pos_2 = -1
        text = tokenizer.tokenize(new_text)
        for j in range(len(text)-4):
            if text[:4] == ['[', 'e', '##1', ']']:
                node_pos_1 = j + 4
            elif text[:4] == ['[', 'e', '##2', ']']:
                node_pos_2 = j + 4

        input_id, token_type_id = inputs["input_ids"], inputs["token_type_ids"]


        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_id)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_id)

        if pad_on_left:
            input_id = ([pad_token] * padding_length) + input_id
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_id = ([pad_token_segment_id] * padding_length) + token_type_id
        else:
            input_id = input_id + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_id = token_type_id + ([pad_token_segment_id] * padding_length)

        assert len(input_id) == max_length, "Error with input length {} vs {}".format(len(input_id), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_id) == max_length, "Error with input length {} vs {}".format(len(token_type_id), max_length)
        #print("IDM",IDM)
        #print("index",all_x[i], all_y[i])
        if IDM[all_x[i],all_y[i]]==2:
            # 2 means generated data, 1 origin data
            sources = 2
        if IDM[all_x[i],all_y[i]]==1:
            sources = 1
        
        #assert IDM[all_x[i],all_y[i]]==1 or IDM[all_x[i],all_y[i]]==2, 'Error with IDM'

        features.append(
            Input_SB_Features(input_ids=input_id,
                        attention_masks=attention_mask,
                        token_type_ids=token_type_id,
                        #matrix=example.matrix,
                        relations=r,
                        doc_id=doc_id,
                        sen_id=sen_id,
                        ids=(all_x[i], all_y[i]),
                        sources = sources  ,
                        node_pos = (node_pos_1, node_pos_2),
                        ))



def build_eval_dataset( rel = None):
    emb = []
    for [j,k,r] in rel:
        if r == 'OVERLAP':
            emb.append([j,k,0])
        if r == 'BEFORE':
            emb.append([j,k,1])
        if r == 'AFTER':
            emb.append([j,k,2])
    return emb

def build_rules(rule = None, rule_tensor = None, n_rule = None):
    all_x, all_y, all_z = np.where(rule_tensor>0)
    n = all_x.shape[0]
    if n==0:
        return [],[]
    rules = [random.randint(0,n-1) for _ in range(n_rule)]
    all_x, all_y, all_z = all_x[rules], all_y[rules], all_z[rules]
    emb = []
    labels = []
    #TODO: think of vector way
    for i in range(n_rule):
        x,y,z = all_x[i],all_y[i], all_z[i]
        emb.extend([torch.tensor([x,y]),torch.tensor([y,z]),torch.tensor([x,z])])
        emb.extend([torch.tensor([y,x]),torch.tensor([z,y]),torch.tensor([z,x])])

    if rule == 'BBB':
        for i in range(n_rule):
            labels.extend([1,1,1])
            labels.extend([2,2,2])
    if rule == 'BOB':
        for i in range(n_rule):
            labels.extend([1,0,1])
            labels.extend([2,0,2])
    if rule == 'OBB':
        for i in range(n_rule):
            labels.extend([0,1,1])
            labels.extend([0,2,2])            
    if rule == 'OOO':
        for i in range(n_rule):
            labels.extend([0,0,0])
            labels.extend([0,0,0])
    
    return emb, labels


def iter_rule_update(BM= None, OM=None,n_iter= 3,  wrong_count=None):
    '''
    iteratively find the ground truth by applying rules
    rules: 
    if Bij Ojk, then Bik
    if Oij Bjk, then Bik
    if Bij Bjk, then Bik
    if Oij Ojk, then Oik
    if Oij, then Oji
    '''
    # first complete OM
    OOM = np.copy(OM)
    OBM = np.copy(BM)
    OM = OM + OM.transpose()
    OM = OM + np.matmul(OM, OM)

    # iteratively update BM
    for _ in range(n_iter):
        BM = BM + np.matmul(BM, OM)
        BM = BM + np.matmul(OM, BM)
        BM = BM + np.matmul(BM, BM)
    
    # normalize
    BM[np.where(BM>0)] = 1
    OM[np.where(OM>0)] = 1

    for i in range(OM.shape[0]):
        OM[i,i] = 0

    if judge_rule(BM):
        wrong_count+=1
        OM = OOM
        BM = OBM
           

    return BM, OM, wrong_count


def rule_tensor(A, B):
    '''
    Cijk = Aij * Bjk
    '''
    n = A.shape[0]
    A = A.reshape(n,n,1)
    B = B.reshape(1,n,n)
    C = A*B
    
    return C

def reverse(r):
    # reverse TLINK
    if r == 0:
        return 0
    if r==1:
        return 2
    if r==2:
        return 1

def reltonum(r):
    # convert rel to num
    if r == 'OVERLAP':
        return 0
    if r=="AFTER":
        return 2
    if r=="BEFORE":
        return 1

def IDIndexDic(rel = None):
    id_list = list(set([r[2] for r in rel]+[r[5] for r in rel]))
    d = {}
    rd = {}
    for i, iid in enumerate(id_list):
        d[iid] = i
        rd[i] = iid
    return d,rd

def build_BO(rel = None, IDToIndex= None):
    n = len(IDToIndex)
    BM = np.zeros((n,n))
    OM = np.zeros((n,n))
    IDM = np.zeros((n,n))
    pos_dict = {}
    for r in rel:
        pos_dict[IDToIndex[r[2]]] = (r[0],r[1])
        pos_dict[IDToIndex[r[5]]] = (r[3],r[4])
        if r[6] == "OVERLAP":
            OM[IDToIndex[r[2]],IDToIndex[r[5]]] = 1
            IDM[IDToIndex[r[2]],IDToIndex[r[5]]] = 1
        if r[6] == "BEFORE":
            BM[IDToIndex[r[2]],IDToIndex[r[5]]] = 1
            IDM[IDToIndex[r[2]],IDToIndex[r[5]]] = 1
        if r[6] == "AFTER":
            BM[IDToIndex[r[5]],IDToIndex[r[2]]] = 1 
            IDM[IDToIndex[r[2]],IDToIndex[r[5]]] = 1
    return BM, OM, IDM, pos_dict


def judge_rule(BM):
    for i in range(BM.shape[0]):
        if BM[i,i] == 1:
            return True
    if np.sum(np.sum(BM* BM.transpose())):
        return True
    return False

glue_tasks_num_labels = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "i2b2-m": 3,
    "i2b2-g": 3,
}

glue_processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    "i2b2-m": I2b2Processor,
    #"i2b2-g": I2b2_Graph_Processor,
    'i2b2-g': I2b2_SB_Processor,
}

glue_output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "i2b2-m": "classification",
    "i2b2-g": "classification",
}


try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix
    from sklearn.metrics import precision_recall_fscore_support as prf
    _has_sklearn = True
except (AttributeError, ImportError) as e:
    logger.warning("To use data.metrics please install scikit-learn. See https://scikit-learn.org/stable/index.html")
    _has_sklearn = False

def is_sklearn_available():
    return _has_sklearn

if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()


    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }

    def p_r_f1(preds, labels):
        f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
        p,r,_,_ = prf(y_true=labels, y_pred=preds, average='micro')
        f2 = f1_score(y_true=labels, y_pred=preds, average='macro')
        p2,r2,_,_ = prf(y_true=labels, y_pred=preds, average='macro')
        conf = confusion_matrix(labels, preds)
        return {
            "micro-precision": p,
            "micro-recall": r,
            "micro-f1": f1,
            "macro-precision": p2,
            "macro-recall": r2,
            "macro-f1": f2,
            "conf": conf
        }


    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }


    def glue_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name == "cola":
            return {"mcc": matthews_corrcoef(labels, preds)}
        elif task_name == "sst-2":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mrpc":
            return acc_and_f1(preds, labels)
        elif task_name == "sts-b":
            return pearson_and_spearman(preds, labels)
        elif task_name == "qqp":
            return acc_and_f1(preds, labels)
        elif task_name == "mnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mnli-mm":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "qnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "rte":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "wnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "i2b2-m":
            return p_r_f1(preds, labels)
        elif task_name == "i2b2-g":
            return p_r_f1(preds, labels)
        else:
            raise KeyError(task_name)
