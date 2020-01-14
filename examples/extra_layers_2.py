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
"""PyTorch BERT model. """


import logging
import math
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import  BertPreTrainedModel, BertModel

# @add_start_docstrings(
#     """Bert Model transformer with a relation classification/regression head on top (a linear layer on top of
#                       the pooled output) e.g. for GLUE tasks. """,
#     BERT_START_DOCSTRING,
#     BERT_INPUTS_DOCSTRING,
# )
class BertForRelationClassification(nn.Module):

    def __init__(self, config):
        super(BertForRelationClassification, self).__init__()
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self, 
        inputs=None, 
        labels=None
    ):

        inputs = self.dropout(inputs)
        logits = self.classifier(inputs)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  


# ## Self-Defined Graph Model
# @add_start_docstrings("""Bert Model transformer with a node embedding head on top (a linear layer on top of
#                      the pooled output) e.g. for I2B2 tasks. """,
#                      BERT_START_DOCSTRING,
#                      BERT_INPUTS_DOCSTRING,
# )
class BertForNodeEmbedding(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForNodeEmbedding, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]
        return pooled_output 



# # @add_start_docstrings("""Bert Model transformer with a relation classification/regression head on top (a linear layer on top of
# #                      the pooled output) e.g. for GLUE tasks. """,
# #                      BERT_START_DOCSTRING,
# #                      BERT_INPUTS_DOCSTRING)
# class BertForRelationClassification(BertPreTrainedModel):

#     def __init__(self, config):
#         super(BertForRelationClassification, self).__init__(config)
#         self.num_labels = config.num_labels
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
#         self.init_weights()

#     def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
#                 position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

#         outputs = self.bert(input_ids,
#                             attention_mask=attention_mask,
#                             token_type_ids=token_type_ids,
#                             position_ids=position_ids,
#                             head_mask=head_mask,
#                             inputs_embeds=inputs_embeds)

#         pooled_output = outputs[1]
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#         outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

#         if labels is not None:
#             if self.num_labels == 1:
#                 #  We are doing regression
#                 loss_fct = MSELoss()
#                 loss = loss_fct(logits.view(-1), labels.view(-1))
#             else:
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             outputs = (loss,) + outputs

#         return outputs  # (loss), logits, (hidden_states), (attentions)

import numpy as np
from torch_geometric.nn import SAGEConv, GATConv
from scipy.sparse import coo_matrix

class ConvGraph(nn.Module):

    def __init__(self, config, GAT=False):
        super(ConvGraph, self).__init__()
        if GAT:
            self.Graphmodel = SAGEConv(config.hidden_size,config.hidden_size) # SAGEConv
        else:
            self.Graphmodel = GATConv(config.hidden_size,config.hidden_size,heads=1) # GAT

    def forward(self, features, adjacency_matrix):

        A = adjacency_matrix
        # transfor A into sparse matrix, to get desired model input
        A_coo = coo_matrix(A)
        row = A_coo.row
        column = A_coo.col
        edge_index = np.asarray([row,column])

        # model input
        edge_index = torch.LongTensor(edge_index)
        outputs = self.Graphmodel(features,edge_index)  

        return outputs



### End