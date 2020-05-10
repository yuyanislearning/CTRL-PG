'''
Created on Apr 7, 2020

@author: Yu Yan

usage: i2b2Evaluation.py  --gold_file_des --system_file_des --data_dir --output_dir

Perform error analysis 
'''

import re
import os
import sys
import argparse
import numpy as np
from utils_relation import glue_processors as processors
from utils_relation import rule_tensor


def load_data(data_path, processor):
    '''
    To load the origin data and find entities that involve in rules by doc_id and sen_id
    '''
    examples = processor.get_test_examples(data_path, tbd = False) 
    features = []
    dict_all = {} # dict_all[doc_id][sen_id]= [entitys and rules]

    is_tf_dataset = False

    
    for (ex_index, example) in enumerate(examples):
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)

        # construct dict to store the transformation from relation id to index of the matrix and vice versa
        IDToIndex, IndexToID = IDIndexDic(rel = example.relations)
        if example.doc_id not in dict_all:
            dict_all[example.doc_id] = {}
        #     dict_all[example.doc_id][example.sen_id] = {}
        # if example.sen_id not in dict_all[example.doc_id]:
        #     dict_all[example.doc_id][example.sen_id] = {}
        
        # find rules
        BM, OM, _, _ = build_BO(rel = example.relations, IDToIndex= IDToIndex)
        BBB, BOB, OBB, OOO = find_rules(BM, OM)
        # if len(seek_rules(BBB, IndexToID))>0:
        #     print(seek_rules(BBB, IndexToID))
        dict_all[example.doc_id][example.sen_id] = (seek_rules(BBB, IndexToID), seek_rules(BOB, IndexToID),  seek_rules(OBB, IndexToID),  seek_rules(OOO, IndexToID))
        # for ID in IDToIndex.keys():
        #     temp = exist_rule(ID, IDToIndex, BBB,BOB,OBB,OOO)
        #     if temp[0]:
        #         dict_all[example.doc_id][example.sen_id][ID] = temp[1]

    return dict_all
    
def seek_rules(BBB, IndexToID):
    '''
    get all the relation and store in double list form
    '''
    x,y,z = np.where(BBB>0)
    rules = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis = 1)
    rules = [[IndexToID[j] for j in i] for i in rules]
    return rules

def exist_rule(ID, IDToIndex, BBB,BOB,OBB,OOO):
    '''
    if ID involve in any rule, if so, return the rules
    '''
    rule_exist = False
    x,y,z = np.where(BBB>0)
    in_BBB = any((IDToIndex[ID] in x, IDToIndex[ID] in y,IDToIndex[ID] in z))
    x,y,z = np.where(BOB>0)
    in_BOB = any((IDToIndex[ID] in x, IDToIndex[ID] in y,IDToIndex[ID] in z))
    x,y,z = np.where(OBB>0)
    in_OBB = any((IDToIndex[ID] in x, IDToIndex[ID] in y,IDToIndex[ID] in z))
    x,y,z = np.where(OOO>0)
    in_OOO = any((IDToIndex[ID] in x, IDToIndex[ID] in y,IDToIndex[ID] in z))
    rule_exist = any(in_BBB, in_OBB, in_BOB, in_OOO)
    

    return rule_exist, [in_BBB, in_BOB, in_OBB, in_OOO]


def find_rules(BM, OM):
    # convert BM to a 3d tensor that BT[:,i,:] = BM for any i 
    BT = np.tile(BM,(BM.shape[0],1,1))
    BT = np.moveaxis(BT, 0,1)
    OT = np.tile(OM,(OM.shape[0],1,1))
    OT = np.moveaxis(OT, 0,1)
    
    # rule tensor find where ij, jk has link, BT find ik has link
    BBB = rule_tensor(BM, BM) * BT
    BOB = rule_tensor(BM, OM) * BT
    OBB = rule_tensor(OM, BM) * BT
    OOO = rule_tensor(OM, OM) * OT

    return BBB, BOB, OBB, OOO


def build_BO(rel = None, IDToIndex= None, tbd = False):
    '''
    convert the relations to before matrix and overlap matrix, not distinguished before and after
    later will construct after matrix by transverse
    '''
    n = len(IDToIndex)
    BM = np.zeros((n,n))
    OM = np.zeros((n,n))
    IDM = np.zeros((n,n))
    pos_dict = {}
    if tbd:
        for r in rel:
            pos_dict[IDToIndex[r[2]]] = (r[0],r[1])
            pos_dict[IDToIndex[r[5]]] = (r[3],r[4])
            if r[6] == "SIMULTANEOUS":
                OM[IDToIndex[r[2]],IDToIndex[r[5]]] = 1
                IDM[IDToIndex[r[2]],IDToIndex[r[5]]] = 1
            if r[6] == "BEFORE":
                BM[IDToIndex[r[2]],IDToIndex[r[5]]] = 1
                IDM[IDToIndex[r[2]],IDToIndex[r[5]]] = 1
            if r[6] == "AFTER":
                BM[IDToIndex[r[5]],IDToIndex[r[2]]] = 1 
                IDM[IDToIndex[r[2]],IDToIndex[r[5]]] = 1

    else:
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

def IDIndexDic(rel = None):
    id_list = list(set([(r[0],r[2]) for r in rel]+[(r[3], r[5]) for r in rel]))
    id_list.sort()
    d = {}
    rd = {}
    for i, iid in enumerate(id_list):
        d[iid[1]] = i
        rd[i] = iid[1]
    return d,rd

def find_error_links(tlinks_gold, tlinks_sys):
    error_links = {}
    gold_links = {}
    for lid in tlinks_sys.keys():
        if tlinks_sys[lid][5] != tlinks_gold[lid][5]:
            error_links[lid] =tlinks_sys[lid]
            gold_links[lid] =tlinks_gold[lid]

    return  gold_links, error_links
            

def attr_by_line(tlinkline, ground_truth):
    """
    Args:
        line - str: MAE TLINK tag line,
                    e.g. <TLINK id="TL70" fromID="E28" fromText="her erythema"
                    toID="E26" toText="erythema on her left leg" type="OVERLAP" />
    """
    if ground_truth:
        re_exp = 'id=\"([^"]*)\"\s+fromID=\"([^"]*)\"\s+fromText=\"([^"]*)\"\s+toID=\"([^"]*)\"\s+toText=\"([^"]*)\"\s+type=\"([^"]*)\"\s+\/>'
    else:
        re_exp = 'id=\"([^"]*)\"\s+fromID=\"([^"]*)\"\s+fromText=\"([^"]*)\"\s+toID=\"([^"]*)\"\s+toText=\"([^"]*)\"\s+type=\"([^"]*)\"\s+senid=\"([^"]*)\"\s+\/>'
    m = re.search(re_exp, tlinkline)
    if m and ground_truth:
        lid, fromid, fromtext, toid, totext, attr_type = m.groups()
        attr_type=attr_type.replace('SIMULTANEOUS','OVERLAP')
        return [lid, fromid, fromtext, toid, totext,  attr_type.upper(), None]
    if m and not ground_truth:
        lid, fromid, fromtext, toid, totext, attr_type, sen_id = m.groups()
    else:
        raise Exception("Malformed EVENT tag: %s" % (tlinkline))
    attr_type=attr_type.replace('SIMULTANEOUS','OVERLAP')
    return [lid, fromid, fromtext, toid, totext,  attr_type.upper(), sen_id]

def get_tlinks(text_fname, ground_truth = False):
    '''
    Args:
        text_fname: file name of the MAE xml file
    
    Output:
        a tlinks tuple of all the tlinks in the file 
    '''
    tf=open(text_fname)
    lines = tf.readlines()
    tlinks={}
    #print(text_fname)
    for line in lines:  
        if re.search('<TLINK',line):
            tlink_list=attr_by_line(line, ground_truth)
            tlinks[tlink_list[1] + tlink_list[3]] = tlink_list
    return tlinks



parser = argparse.ArgumentParser(description='Evaluate system output against gold standard.')
parser.add_argument('--gold_file_des', type=str, nargs=1,\
                    help='the file or directory of the gold standard xml file/directory')
parser.add_argument('--system_file_des', type=str, nargs=1,\
                    help='the file or directory of the system output xml file/directory')
parser.add_argument('--data_path', type=str,\
                    help='the file or directory of the data')
parser.add_argument('--output_dir', type=str, \
                    help='the directory to store the output')
args = parser.parse_args()

# get the data from cached
processor = processors['i2b2-g']()
dict_all = load_data(args.data_path, processor)


goldDir=args.gold_file_des[0]
systemDir=args.system_file_des[0]
if os.path.isdir(goldDir+'/') and goldDir[-1] != '/':
    goldDir+='/'
if os.path.isdir(systemDir+'/') and systemDir[-1]!="/":
    systemDir+='/'

if os.path.isdir(goldDir) and os.path.isdir(systemDir):
    goldFileList=os.listdir(goldDir)
    systemFileList=os.listdir(systemDir)
    for fle in goldFileList:
        doc_id = fle[0:len(fle)-4]
        f = open(args.output_dir + doc_id+'.txt', 'w')
        tlinks_gold=get_tlinks(os.path.join(goldDir,fle), ground_truth = True)
        tlinks_sys=get_tlinks(os.path.join(systemDir,fle))
        gold_links, error_links = find_error_links(tlinks_gold, tlinks_sys)


        for lid in error_links.keys():
            sen_id = error_links[lid][6]
            
            e1, e2 = error_links[lid][1], error_links[lid][3]
            f.write('e1 id: ' + e1 + '; e2 id: ' + e2 + '; e1 text: '+ error_links[lid][2]+ '; e2 text: ' + error_links[lid][4] + '\n')
            f.write('ground truth: ' + gold_links[lid][5] + '; prediction: ' + error_links[lid][5] + '\n')
            sen_id = '(' + str(int(sen_id)-1) + ', ' + sen_id + ', ' + str(int(sen_id)+1) + ')'
            if sen_id in dict_all[int(doc_id)]:
                
                BBB_rule, BOB_rule, OBB_rule, OOO_rule = dict_all[int(doc_id)][sen_id]
                # print(e1,e2)
                # print(BBB_rule)
                if  any([e1 in r and e2 in r for r in BBB_rule]):
                    f.write('This relation involves in BBB\n')
                    #print(e1,e2, BBB_rule)
                if  any([e1 in r and e2 in r for r in BOB_rule]):
                    f.write('This relation involves in BOB\n')
                    #print(e1,e2, BOB_rule)
                if  any([e1 in r and e2 in r for r in OBB_rule]):
                    f.write('This relation involves in OBB\n')
                    #print(e1,e2, OBB_rule)
                if  any([e1 in r and e2 in r for r in OOO_rule]):
                    f.write('This relation involves in OOO\n')
                    #print(e1,e2, OOO_rule)
            f.write('\n')
        f.close()
                


