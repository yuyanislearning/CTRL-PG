import json
json_file = 'event-time.json'
with open(json_file) as f2:
    dick = json.load(f2)


f = open('doc_list.txt', 'r')
fw = open('distance.txt','w')


for line in f:
    line = line.strip()
    doc_id = line[0:len(line)-4]
    with open("/tmp/TIME/aug_0psl_0.0_%s.output.txt"%(doc_id)) as f3:
        for li in f3:
            [e1, e2, pred, p] = li.strip().split('\t')
            d = abs(int(dick[doc_id][e1]['pos'][0]) - int(dick[doc_id][e2]['pos'][0]))
            fw.write('\t'.join([e1[0], e2[0], pred, p, str(d)]) + '\n')


f.close()
fw.close()
            
            

