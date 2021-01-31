from __future__ import absolute_import, division, print_function
import argparse
import glob
import logging
import os, sys
import random
import sys
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from closure import evaluation as closure_evaluate
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer,
                                  DistilBertConfig,
                                  DistilBertForSequenceClassification,
                                  DistilBertTokenizer,
                                  AlbertConfig,
                                  AlbertForSequenceClassification, 
                                  AlbertTokenizer,
                                )
from model_layers import BertForRelationClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from utils_relation import glue_compute_metrics as compute_metrics
from utils_relation import glue_output_modes as output_modes
from utils_relation import glue_processors as processors
from utils_relation import sb_convert_examples_to_features as convert_examples_to_features


logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, 
                                                                                RobertaConfig, DistilBertConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForRelationClassification, BertTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, dict_IndenToID, label_dict):
    """ Train the model """
    # keep track of the best f1
    best_mif1, best_maf1 = 0, 0
    best_check = None
    best_f1s = []

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    #results = evaluate(args, model, tokenizer)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # convert the example from three cases one example to one case one exsample
            batch = tuple(t.view(-1).to(args.device) if len(t.size()) ==2 else t.view(t.size()[0]*t.size()[1],-1).to(args.device) for t in batch) 
            class_weights = args.class_weight.split('~')
            inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'labels':         batch[4],
                        'rules':          batch[7],
                        'psllda':         args.psllda,
                        'class_weights':  class_weights,
                        }


            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        best_mif1, best_maf1, best_check, results = evaluate(best_mif1, best_maf1,best_check,global_step, args, model, tokenizer, final_evaluate = False)
                        best_f1s.append((global_step,best_mif1))

                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    tokenizer.save_pretrained(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)
                
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    print("micro-f1", best_mif1)
    print("macro-f1", best_maf1)
    print("best checkpoint", best_check)
    print("best f1s", best_f1s)
    best_f1s = np.array(best_f1s)

    return global_step, tr_loss / global_step, best_check


def evaluate(best_mif1, best_maf1, best_check, check,  args, model, tokenizer,  prefix="", final_evaluate = False):
    '''
    evaluate on the dev or test data, update best f1 score 
    '''
    softmax = torch.nn.Softmax(dim=1)
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset, dict_IndenToID, label_dict = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True, final_evaluate = final_evaluate)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        events = None
        doc_ids = None
        sent_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()

            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                            'attention_mask': batch[1],
                            'labels':         batch[4],
                            'evaluate': True,
                            'psllda':         args.psllda,
                            }

                event_ids = batch[3]
                document_ids = batch[5]
                sentence_ids = batch[6]
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs) 
                tmp_eval_loss, logits = outputs[:2]

                if args.tbd:
                    eval_loss = 0
                else:
                    eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

            if preds is None:
                preds = softmax(logits).detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, softmax(logits).detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

            if events is None:
                events = event_ids.detach().cpu().numpy()
            else:
                events = np.append(events, event_ids.detach().cpu().numpy(), axis=0)

            if doc_ids is None:
                doc_ids = document_ids.detach().cpu().numpy()
            else:
                doc_ids = np.append(doc_ids, document_ids.detach().cpu().numpy(), axis=0)

            if sent_ids is None:
                sent_ids = sentence_ids.detach().cpu().numpy()
            else:
                sent_ids = np.append(sent_ids, sentence_ids.detach().cpu().numpy(), axis=0)


        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            labels = np.argmax(preds, axis=1)
            if not args.tbd:
                preds = np.max(preds, axis = 1)

        elif args.output_mode == "regression":
            labels = np.squeeze(preds)
            preds = np.max(preds, axis = 1)

        result = compute_metrics(eval_task, labels , out_label_ids)
        results.update(result)

        if best_mif1 < results['micro-f1']:
            best_mif1 = results['micro-f1']
            best_maf1 = results['macro-f1']
            best_check = check
            print("best_mif1",best_mif1)
            print("best_maf1",best_maf1)
            print("best_check",best_check)
            
        output_eval_file = os.path.join(eval_output_dir, prefix,  "eval_results" + '_aug' + str(args.aug_round) + "_psl" + str(args.psllda) + ".txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        if True:
            doc_dict = {}
            for doc_id, sen_id, label, event, pred in zip(doc_ids, sent_ids, labels, events, preds):

                if args.tbd:
                    doc_id = str(doc_id).zfill(4)

                if doc_id not in doc_dict:
                    doc_dict[doc_id] = {"labels":[], "events":[], "sen_ids":[], 'preds':[]}
                if args.tbd:

                    event = [dict_IndenToID[str(doc_id)+"[" + str(sen_id[0])+":"+str(sen_id[1]) + ")"][x] for x in event]
                else:
                    event = [dict_IndenToID[str(doc_id)+"(" + str(sen_id[0])+", "+str(sen_id[1]) +", " +str(sen_id[2]) + ")"][x] for x in event]
                doc_dict[doc_id]["preds"].append(pred)
                doc_dict[doc_id]["events"].append(event)
                doc_dict[doc_id]["sen_ids"].append(sen_id)
                doc_dict[doc_id]['labels'].append(label)
            
            for doc_id in doc_dict.keys():
                labels = doc_dict[doc_id]["labels"]
                if args.tbd:
                    temp_label_dict =  {0: 'overlap', 1: 'before', 2: 'after', 3:'vague', 4:'includs', 5:'is_included'}
                    labels = [temp_label_dict[x] for x in labels]
                else:
                    labels = [label_dict[x] for x in labels]
                events = doc_dict[doc_id]["events"]
                sen_ids = doc_dict[doc_id]["sen_ids"]

                if final_evaluate:
                    if args.tempeval:
                        ce = closure_evaluate(doc_id, args.final_xml_folder)
                        ce.eval(labels, events, sen_ids, preds)
                    fw = open(args.output_dir + 'aug_'+ str(args.aug_round)+'psl_' + str(args.psllda) +'_'+str(doc_id)+ '.output.txt', 'w')
                    for [id1, id2], label,  pred in zip(events, labels,  preds):
                        fw.write('\t'.join([str(id1),str(id2),label, str(pred)]) + '\n')
                    fw.close()


                else:
                    if args.tempeval:
                        ce = closure_evaluate(doc_id, args.xml_folder)
                        ce.eval(labels, events, sen_ids, preds)
                    # for test
                    fw = open(args.output_dir + 'aug_'+ str(args.aug_round)+'psl_' + str(args.psllda) +'_'+str(doc_id)+ '.output.txt', 'w')
                    for [id1, id2], label,  pred in zip(events, labels,  preds):
                        fw.write('\t'.join([str(id1),str(id2),label, str(pred)]) + '\n')
                    fw.close()
            if final_evaluate and args.tbd:
                os.system('for thres in 0.1 0.3 0.5 0.7 0.9 ; do python3 vague_processing.py $thres; done')
            if final_evaluate and args.tempeval:# temporal evaluation
                os.system(' '.join(["python2 i2b2-evaluate/i2b2Evaluation.py --tempeval",str(args.test_gold_file),str(args.final_xml_folder)]) + ' > ' + args.output_dir + 'aug_' + str(args.aug_round) + '_psl_'+ str(args.psllda) + '_closure_results.txt')
                os.system(' '.join(["python2 i2b2-evaluate/i2b2Evaluation.py --tempeval",str(args.gold_file),str(args.xml_folder)]))

    return best_mif1, best_maf1, best_check,results


def load_and_cache_examples(args, task, tokenizer, evaluate=False, final_evaluate = False):
    '''
    To load and cache examples
    if final_evaluate == T, it will load test data
    else if evaluate == T, it will load dev data
    else it will load training data
    '''
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_{}'.format(
        'test' if final_evaluate else 'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task),
        str(args.aug_round)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not evaluate: 
        # load cache if exists
        logger.info("Loading features from cached file %s", cached_features_file)
        features,dict_IndenToID = torch.load(cached_features_file)
        label_list = processor.get_labels(args.tbd)
        label_dict = {x:y for x,y in enumerate(label_list)}
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels(args.tbd)
        if not args.tbd:
            label_dict = {x:y for x,y in enumerate(label_list)}
        else:
            label_dict = {0: 'overlap', 1: 'before', 2: 'after'}#,  3:'includs', 4:'is_included'} TODO
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1] 
        # final_evaluate indicate test data; only evaluate indicate dev data
        examples = processor.get_test_examples(args.data_dir, args.tbd) if final_evaluate else processor.get_dev_examples(args.data_dir, args.tbd) if evaluate else processor.get_train_examples(args.data_dir, args.tbd)

        features, dict_IndenToID = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=args.max_seq_length,
                                                output_mode=output_mode,
                                                pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                data_aug = args.data_aug,
                                                evaluate = evaluate,
                                                aug_round = args.aug_round,
                                                tbd = args.tbd,
                                                acrobat = args.acrobat,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save((features,dict_IndenToID), cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_masks for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_event_ids = torch.tensor([f.ids for f in features], dtype=torch.long)
    #all_node_pos = torch.tensor([f.node_pos for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.relations for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.relations for f in features], dtype=torch.float)
    all_doc_ids = torch.tensor([f.doc_id for f in features], dtype = torch.int)
    all_sources = [f.sources for f in features] 
    if not evaluate:
        all_rules = torch.tensor([f.rules for f in features], dtype = torch.int)
        if not args.tbd:
            all_sen_ids = torch.tensor([[[int(i) for i in s[1:len(s)-1].replace(':',', ').split(", ")] for s in f.sen_id] for f in features])
        else:
            all_sen_ids = torch.tensor([[[int(i) for i in s[1:len(s)-1].split(":")] for s in f.sen_id] for f in features])    


        data_type = None

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_event_ids, all_labels, all_doc_ids, all_sen_ids,  all_rules)#,all_node_pos , all_event_ids)
    else:
        if not args.tbd:
            all_sen_ids = torch.tensor([[int(i) for i in f.sen_id[1:len(f.sen_id)-1].replace(':',', ').split(", ")] for f in features])
        else:
            all_sen_ids = torch.tensor([[int(i) for i in f.sen_id[1:len(f.sen_id)-1].split(":")] for f in features])    
        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_event_ids, all_labels, all_doc_ids, all_sen_ids)#,all_node_pos , all_event_ids)
    return dataset, dict_IndenToID, label_dict



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--gold_file", default="glue_data/I2B2-R/ground-truth/dev/merged_xml", type=str, 
                        help="The gold standard file. ")
    parser.add_argument("--test_gold_file", default="glue_data/I2B2-R/ground-truth/dev/merged_xml", type=str, 
                        help="The test set gold standard file. ")
    parser.add_argument("--xml_folder", default="glue_data/I2B2-R/rich_relation_dataset_2/merged_xml/3/dev-empty/", type=str,
                        help="The xml data dir to put result for dev set. ")
    parser.add_argument("--final_xml_folder", default="glue_data/I2B2-R/rich_relation_dataset_2/merged_xml/3/test-empty/", type=str, 
                        help="The xml data dir to put result for test set. ")                        
    parser.add_argument("--model_type", default='bert', type=str, 
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str, 
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--task_name", default='i2b2-g', type=str, 
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--error_output_dir", default=None, type=str,
                        help="The output directory where error analysis will be written.")


    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--data_aug", default='triple_rules', type=str,
                        help="Whether to run data augmentation.")
    parser.add_argument("--class_weight", default='1~1~1', type=str,
                        help="class weights of the three classes: overlap; before and after.")
    
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--tempeval", action='store_true',
                        help="Set this flag if you are using a temporal evaluation.")
    parser.add_argument("--tbd", action='store_true',
                        help="Set this flag if you are using a TBDense data.")
    parser.add_argument("--acrobat", action='store_true',
                        help="Set this flag if you are using a ACROBAT data.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")     
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--psllda', type=float, default=0,
                        help="lambda for controlling soft probabilistic logit loss")

    parser.add_argument('--logging_steps', type=int, default=500,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--aug_round', type=int, default=0,
                        help="augment data for X round")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--node_embed", action='store_true',
                        help="node embedding")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    # parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    # parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels(args.tbd)
    num_labels = len(label_list)
    label_dict = {i:x for i,x in enumerate(label_list)}

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          finetuning_task=args.task_name,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)


    # Training
    if args.do_train:
        #results = evaluate(args, model, tokenizer )
        train_dataset, dict_IndenToID, label_dict = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss, best_check = train(args, train_dataset, model, tokenizer, dict_IndenToID, label_dict)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model = model_class.from_pretrained(os.path.join(args.output_dir, 'checkpoint-{}'.format(best_check)))
        tokenizer = tokenizer_class.from_pretrained(os.path.join(args.output_dir, 'checkpoint-{}'.format(best_check)))

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        model.to(args.device)

    # Evaluation
    results = {} 
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        _,_,_, result = evaluate(0,0,0,0,args, model, tokenizer,  final_evaluate =True)
        results.update(result)
        # get the dev TODO remove it
        _,_,_, result = evaluate(0,0,0,0,args, model, tokenizer,  final_evaluate =False)
        results.update(result)


if __name__ == "__main__":
    main()
