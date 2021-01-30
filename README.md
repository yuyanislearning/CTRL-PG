# Examples

In this section a few examples are put together. 



## Examples

To run temporal relation extraction


### TBDense dataset:
```bash

python run_rich_relation.py
     --do_train \
     --do_eval \
     --tbd \
     --evaluate_during_training \
     --do_lower_case \
     --data_dir ../tbd/all_context/ \
     --max_seq_length 128 \
     --per_gpu_eval_batch_size=8 \
     --per_gpu_train_batch_size=8 \
     --learning_rate 2e-5 \
     --num_train_epochs 10.0 \
     --output_dir /tmp/tbd \
     --overwrite_output_dir \
     --data_aug triple_rules \
     --aug_round 0 \
     --psllda 0

```
To perform Global inference, two xml file will need for both test and dev data. 
Gold file should contain the ground truth information of temporal relation. An example is shown below:
`<TLINK id="TL23" fromID="E107" fromText="a small ventral hernia" toID="E30" toText="This" type="OVERLAP" />`
TLINK id is the id of the temporal relation; fromID and toID are the entity IDs; fromText and toText are Text associated with the entity; type is the relation.
Similarly, an template xml file is required to write the output, which should have the ID and text information and leave blank for type. 
--gold_file is the path to dev ground true xml; --xml_folder is the path to dev result xml; --final_xml_folder is the path to test result xml; --test_gold_file is the path to test result xml.
--psllda is the parameter that controls the weight of probabilistic soft logic loss introduced.







