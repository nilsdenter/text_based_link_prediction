import csv
import pandas
from time import time

Start = 1
End = 3

for i in range(Start,End+1,1):
    starttime = time()
    csv_file = "tdm_4years_wl4_n2_w4.csv"
    term_constellation=[]
    vocab = []
    vocab_all = []
    
    with open(csv_file, "r") as tdm:
        csvreader = csv.reader(tdm, delimiter=';')
        next(tdm)
        
        for row in csvreader:
            vocab_all.append(row[0].split(' ')[0])
            vocab_all.append(row[0].split(' ')[1])
            if row[(i-(Start-1))] == "":  
                vocab.append(row[0].split(' ')[0])
                vocab.append(row[0].split(' ')[1])
            
            elif row[0].split(' ')[0] != row[0].split(' ')[1]:
                if row[(i-(Start-1))] in (None, ""):
                    continue
                else:
                   for k in range(0,int(row[(i-(Start-1))])):
                       term_constellation.append([row[0].split(' ')[0], row[0].split(' ')[1]])
              
    
    vocab_all = list(dict.fromkeys(vocab_all))
    vocab_all.sort()
    
    vocab = list(dict.fromkeys(vocab))
    vocab.sort()
    
    
    import itertools
    import collections
    
    result = collections.defaultdict(lambda: collections.defaultdict(int))
    
    for row in term_constellation:
        counts = collections.Counter(row)
        for key_from, key_to in itertools.permutations(counts, 2):
            result[key_from][key_to] += counts[key_from] * counts[key_to]
            
    df = pandas.DataFrame(data=result)
    df = df.reindex(index=df.index.union(vocab), columns= df.columns.union(vocab))
    pandas.DataFrame(data=df).to_csv("wl4_n2_w4_t%i_allyears_tf_cooccurence.csv" %(i), sep=";", decimal=",")
   
