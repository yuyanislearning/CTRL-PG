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
import torch.nn.functional as F

from transformers import  BertPreTrainedModel, BertModel, BertForSequenceClassification

import numpy as np
from torch_geometric.nn import SAGEConv, GATConv
from scipy.sparse import coo_matrix



class GraphConvClassification(nn.Module):

    def __init__(self, config, GAT = False):
        super(GraphConvClassification, self).__init__()
        self.dim_emb = 768
        #self.dim_emb = config.hidden_size
        self.linear = nn.Linear(config.hidden_size, self.dim_emb)
        self.linear2 = nn.Linear(self.dim_emb, self.dim_emb)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.dim_emb*2, config.num_labels)
        #self.classifier = nn.Linear(config.hidden_size*2, config.num_labels)
        if not GAT:
            self.Graphmodel = SAGEConv(self.dim_emb,self.dim_emb,aggr = "mean")
            #self.Graphmodel = SAGEConv(config.hidden_size,config.hidden_size) # SAGEConv
        else:
            self.Graphmodel = GATConv(self.dim_emb,self.dim_emb,heads=1) # GAT


    def forward(
        self, idx= None, adjacency_matrix=None, node_embeddings = None,
        label=None
    ):
        """inputs could be (doc_size, number_node_pair,2*embeding_size)"""

        #TODO: remove to outside
        A = adjacency_matrix
        # transfor A into sparse matrix, to get desired model input
        A_coo = coo_matrix(A)
        row = A_coo.row
        column = A_coo.col
        edge_index = np.asarray([row,column])

        # model input
        edge_index = torch.LongTensor(edge_index).cuda()
        #print("number of links: ",edge_index.size())
        node_embeddings = F.relu(self.linear(node_embeddings))
        node_out = self.Graphmodel(node_embeddings,edge_index)  
        # residual block
        node_embeddings = node_out + node_embeddings
        # second layer of graph
        #node_out = self.Graphmodel(node_embeddings,edge_index) 
        #node_embeddings = node_out + node_embeddings

        # select nodes to be classify
        outputs = torch.cat((node_embeddings[idx[:, 0]], node_embeddings[idx[:,1]]), dim = 1)
        inputs = self.dropout(outputs)
        logits = self.classifier(inputs)
        outputs = logits

        if label is not None:

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
            #outputs = (loss,) + outputs

        loss = loss + PSL_loss(label, logits)


        return (loss, outputs)  

class NoGraphClassification(nn.Module):

    def __init__(self, config, cal_hidden_loss = True):
        super(NoGraphClassification, self).__init__()
        self.num_labels = config.num_labels
        self.dim_emb = 768
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.dim_emb, config.num_labels)
        self.hidden_classifier = nn.Linear(self.dim_emb*2, config.num_labels)
        #self.cal_hidden_loss = cal_hidden_loss
        #self.classifier = nn.Linear(config.hidden_size*2, config.num_labels)

    def forward(
        self, adjacency_matrix = None, idx = None, node_embeddings = None,
        label=None, cal_hidden_loss = True, weight =0.3
    ):
        """inputs could be (doc_size, number_node_pair,2*embeding_size)"""

        inputs = self.dropout(node_embeddings)
        logits = self.classifier(inputs)
        outputs = logits

        if label is not None:

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), label.view(-1))
            #outputs = (loss,) + outputs
        
        #print(label)

        if cal_hidden_loss:
            hidden_loss = torch.tensor(0, dtype = torch.float64).cuda()
            for n_batch in range(int(label.size()[0]/2)):
                #print(node_embeddings.size())
                #print(label.size())
                #print(node_embeddings[n_batch*2,:].size())
                hidden_inputs = torch.cat((node_embeddings[n_batch*2,:], node_embeddings[n_batch*2+1,:]))
                hidden_inputs = self.dropout(hidden_inputs)
                hidden_logits = self.hidden_classifier(hidden_inputs)
                hidden_label = torch.tensor(identify_label(label[n_batch*2], label[n_batch*2+1]))
                loss_fct = CrossEntropyLoss()
                hidden_loss = hidden_loss + loss_fct(hidden_logits.view(-1, self.num_labels), hidden_label.view(-1).cuda())
            loss = loss + weight * hidden_loss

        return (loss, outputs)  



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


def PSL_loss( logits=None, rules = None, stick_rule = True, loss = None):
    '''
    PSL loss, fixing true label to calculate the loss
    '''
    psl_loss = torch.tensor(0, dtype = torch.float64).cuda()

    relations_dict = {
            1: [1,1,1],
            2: [2,2,2],
            3: [1,0,1],
            4: [0,2,2],
            5: [0,1,1],
            6: [2,0,2],
            7: [0,0,0],
        }
    s = nn.Softmax(1)
    logits = s(logits)
    
    for n_batch in range(int(logits.size()[0]/3)):
        rule = rules[n_batch]
        if rule == 0:# no rule exists
            continue
        relation = relations_dict[int(rule)]
        i = logits[n_batch*3+0,relation[0]]
        j = logits[n_batch*3+1,relation[1]]
        k = logits[n_batch*3+2,relation[2]]
        psl_loss = psl_loss + max(0, max(0,i+j-1)-k)

    
    return psl_loss

def identify_label(label1 = None, label2 = None):
    ruleB = [(1,1),(1,0),(0,1)]
    ruleO = [(0,0)]
    ruleA = [(2,2),(2,0),(0,2)]
    #print(label1, label2)
    if any(nd==(label1, label2) for nd in ruleB):
        label = 1
    if any(nd==(label1, label2) for nd in ruleO):
        label = 0
    if any(nd==(label1, label2) for nd in ruleA):
        label = 2

    return label



class BertForRelationClassification(BertPreTrainedModel):
    '''
    for relation classification with psl loss
    '''
    def __init__(self, config):
        super().__init__(config)
        #TODO change the num 
        self.num_labels = config.num_labels + 3

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # TODO change the number of labels
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels+3)
        #self.converter = nn.Linear(2*config.hidden_size, config.hidden_size)
        #self.hidden_size = config.hidden_size

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, node_pos_ids=None, psllda = None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None, rules = None, evaluate = False, class_weights = [1,1,1,1,1,1]):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1] # (8, 768)

        # shape of outputs[0]: (8,128,768) -> (batch_size, sequence_length, embedding_size)
        # shape of node_pos_ids: (8, 2, 128)
        #print(node_pos_ids)
        '''
        if node_pos_ids is not None:

            node_embedding = torch.matmul(outputs[0], node_pos_ids) # (8, 2, 768)
            node_embedding = node_embedding.view(-1, 2*self.hidden_size) # (8, 1536)
            pooled_output = self.converter(node_embedding)
            # pooled_output += self.converter(node_embedding)
        '''

        # for class imbalanced
        
        class_weights = torch.tensor([float(cw) for cw in class_weights]).cuda()
 
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here


        psl_loss = True
        lda = psllda
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=class_weights)#) 
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            if psl_loss and not evaluate:
                loss = loss  + lda * PSL_loss( logits=logits, rules = rules,loss = loss)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

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

        return outputs  # (loss), logits, (hidden_states), (attentions)

### End
