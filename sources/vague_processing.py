import os
import re
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import sys


name_list = [fle.strip() for fle in os.popen('cd /tmp/tbd/; ls aug_0psl_%s_*.output.txt'%(0.0))]

def KLdivergence(logits):
    return np.sum(np.array([i*np.log(5*i) for i in logits]))

label_dict = {0: 'overlap', 1: 'before', 2: 'after', 3:'includs', 4:'is_included'}

y_true = []
y_pred = []

threshold = float(sys.argv[1])
for name in name_list:
    with open('/tmp/tbd/%s'%(name)) as f:
        for line in f:
            try:
                line = line.strip().split('\t')
                label = line[2]
                temp = line[3][1:len(line[3])-1].split(' ')
                logits = [float(i) for i in temp if len(i)>0]
                if len(logits)>5:
                    print(logits)
                kl = KLdivergence(logits)
                predict = label_dict[np.argmax(np.array(logits))]
                if kl < threshold:
                    predict = 'vague'
                
                y_true.append(label)
                y_pred.append(predict)
            except:
                continue

print('micro f1 score:',f1_score(y_true, y_pred, average = 'micro'))
print(confusion_matrix(y_true, y_pred))

            
            
            
              


