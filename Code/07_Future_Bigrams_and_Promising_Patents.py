import pandas as pd
import regex
from nltk.util import skipgrams
from time import time
t0 = time()
file_text_t3 = "t3.csv"

promising_bigrams = set(pd.read_csv("Future_bigrams.csv", header = 0, index_col= 0).index)

terms_len = []
for i in promising_bigrams:
    tpl = i.split()
    if tpl[0] == tpl[1]:
        print("duplicate")
    for word in tpl:
        terms_len.append(len(word))
print(min(terms_len))
print(max(terms_len))
    
"""Read csv file"""
df = pd.read_csv(file_text_t3, sep='\t', header = 0, index_col = 0,  lineterminator='\r').dropna()

promising_patents = []
number_bigrams_list = []
number_unique_bigrams_list = []
unique_bigrams_list = []
number_possible_unique_bigrams = []
unqiue_ratio_promising_to_possible = []
counter = 1
t1 = time()

for patent, row in df.iterrows():
    patent_raw = patent
    patent = patent.split()[1]
    patent = patent.split("#")[1]
    patent = patent.split(",")
    patent = "".join(patent)
    texts_raw = row["Title"] + " " + row["Abstract"] + " " + row["Claims"]
    texts_raw = regex.sub(r'\P{L}+', ' ', texts_raw)
    texts_raw.replace("."," ")
    texts_raw.replace("€"," ")
    texts_raw.replace("$"," ")
    texts_raw.replace(","," ") 
    texts_raw.replace("!"," ") 
    texts_raw.replace("?"," ") 
    texts_raw.replace("&"," ") 
    texts_raw.replace("-"," ")
    texts_raw.replace(";"," ")
    texts_raw.replace("/"," ")
    texts_raw.replace(")"," ") 
    texts_raw.replace("("," ")
    texts_raw.replace("+"," ") 
    texts_raw.replace("="," ")
    texts_raw.replace("\\"," ") 
    texts_raw.replace(":"," ")
    texts_raw.replace("'"," ") 
    texts_raw.replace("`"," ") 
    texts_raw.replace("´"," ")
    texts_raw.replace("#"," ")
    texts_raw.replace("%"," ")
    texts_raw.replace("§"," ")
    texts_raw.replace("ß"," ")
    
    #tokenize and lowercase
    texts_raw = texts_raw.lower().split()
    texts_raw = [word for word in texts_raw if len(word) > 3]
    texts_raw = set(texts_raw)
    bigrams_raw = set(skipgrams(texts_raw,2,len(texts_raw)*2))
    bigrams = []
    for word in bigrams_raw: 
        l = [word[0], word[1]]
        l.sort()
        bigrams.append(str(l[0]) + " " + str(l[1]))
        
    number_unique_bigrams = 0
    unique_bigrams = []
    
    bigrams = set(bigrams)
    for promising_bigram in promising_bigrams:
        if promising_bigram in bigrams:
            number_unique_bigrams += 1
            unique_bigrams.append(promising_bigram)
    
    unique_bigrams.sort()
    unique_bigrams_2 = ""
    for bigram in unique_bigrams:
        if len(unique_bigrams_2) == 0:
            unique_bigrams_2 += bigram
        else:
            unique_bigrams_2 += ", "
            unique_bigrams_2 += bigram
    promising_patents.append(patent)
    number_possible_unique_bigrams.append(len(bigrams))
    number_unique_bigrams_list.append(number_unique_bigrams)
    unique_bigrams_list.append(unique_bigrams_2)
    unqiue_ratio_promising_to_possible.append(round(number_unique_bigrams/len(bigrams),4))
    if counter%500==0:
        print("Next 500 patents processed in {0} seconds ({1} of {2}).".format(int(time()-t1), counter, len(df)))
        t1 = time()
    counter += 1

output = pd.DataFrame()
output["promising patents"] = promising_patents
output["# unique bigrams"] = number_unique_bigrams_list
output["# possible unique bigrams"] = number_possible_unique_bigrams
output["ratio"] = unqiue_ratio_promising_to_possible
output["unique bigrams"] = unique_bigrams_list
output.to_excel("Promisingness_score_t3_sorted.xlsx", index=False)
print("\nFinished in {0} seconds.".format(int(time()-t0)))