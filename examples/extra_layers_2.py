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

import numpy as np
from torch_geometric.nn import SAGEConv, GATConv
from scipy.sparse import coo_matrix



class BertForRelationClassification(nn.Module):

    def __init__(self, config):
        super(BertForRelationClassification, self).__init__()
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*2, config.num_labels)

    def forward(
        self, 
        inputs=None, 
        label=None
    ):
        """inputs could be (doc_size, number_node_pair,2*embeding_size)"""

        inputs = self.dropout(inputs)
        outputs = self.classifier(inputs)

        if label is not None:

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
            outputs = (loss,) + outputs

        return outputs  



class BertForNodeEmbedding(BertPreTrainedModel):

    def __init__(self, config):
        super(BertForNodeEmbedding, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.weight = nn.Linear(config.hidden_size, 100)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None ):
        input_ids = torch.squeeze(input_ids, 1)
        attention_mask = torch.squeeze(attention_mask, 1)
        token_type_ids = torch.squeeze(token_type_ids, 1)

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        # outputs[0] is loss; outputs[1] is pooled output
        # should be ( node_size, (1), feature_size)


        pooled_output = outputs[1]
        
        return pooled_output 
    


class ConvGraph(nn.Module):

    def __init__(self, config, GAT=False):
        super(ConvGraph, self).__init__()
        if not GAT:
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
        edge_index = torch.LongTensor(edge_index).cuda()
        outputs = self.Graphmodel(features,edge_index)  

        return outputs



### End
