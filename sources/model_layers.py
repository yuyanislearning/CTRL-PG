import logging
import math
import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from transformers import  BertPreTrainedModel, BertModel
import numpy as np


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
        self.num_labels = config.num_labels 
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)#TODO +2
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, node_pos_ids=None, psllda = None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None, rules = None, evaluate = False, class_weights = [1,1,1,1,1]):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1] # (8, 768)

        # for class imbalanced
        class_weights = torch.tensor([float(cw) for cw in class_weights]).cuda()
 
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here


        psl_loss = True
        lda = psllda
        if labels is not None: #and not evaluate: TODO
            loss_fct = CrossEntropyLoss()#weight=class_weights) 
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            if psl_loss and not evaluate:
                loss = loss  + lda * PSL_loss(logits=logits, rules = rules,loss = loss)
            outputs = (loss,) + outputs


        return outputs  # (loss), logits, (hidden_states), (attentions)

