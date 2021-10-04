import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from time import time
t0 = time()
import nltk
nltk.download('punkt')
import numpy as np
import pandas as pd

anzahl_terme = 1000

window_size_bigrams = 4

Start = 1
End = 3
dimensions= 300
window_size=5
min_count=1
noise_terms=15
file = "wl4_n2_w%i_t1_4years_tf_cooccurence.csv" %window_size_bigrams

df = pd.read_csv(file, sep=";", decimal=",")
unigrams = df.iloc[:,[0]].values.tolist()

i = 0
number_bigrams = 0
bigrams = []
for unigram1 in unigrams:
    l=0
    for unigram2 in unigrams[(i+1):]:
        l+=1
        if str(unigram2)=="[nan]":
            unigram2 = "null"
            unigram1 = "".join(unigram1)
        elif str(unigram1)=="[nan]":
            unigram1 = "null"
            unigram2 = "".join(unigram2)
        else:
            unigram1 = "".join(unigram1)
            unigram2 = "".join(unigram2)
        if unigram1 != unigram2:
            bigrams.append(unigram1 + " "+ unigram2)
            number_bigrams += 1
    i += 1
    
print("\nIn total %i bigrams generated!\n" %(number_bigrams))

pd.DataFrame(bigrams).to_csv("%i_bigrams.csv" %number_bigrams, sep=";", header=["Bigram"])
len_texts = []
for time_slice in range(Start, End+1):
    
    """Load in data"""
    #text_files = "textfiles/"
    if time_slice==1:
        csv_file = "t1.csv"
    if time_slice==2:
        csv_file = "t2.csv"
    if time_slice==3:
        csv_file = "t3.csv"
    
    """Read csv file"""
    df = pd.read_csv(csv_file, sep='\t', header = 0, index_col = 0,  lineterminator='\r').dropna()
    texts_raw = df["Title"] + " "+ df["Abstract"] +" " + df["Claims"]
    texts=[''.join([i for i in str(s) if not i.isdigit()]) for s in texts_raw]
    texts=[i.replace("."," ") for i in texts]
    texts=[i.replace(","," ") for i in texts]
    texts=[i.replace("!"," ") for i in texts]
    texts=[i.replace("?"," ") for i in texts]
    texts=[i.replace("&"," ") for i in texts]
    texts=[i.replace("-"," ") for i in texts]
    texts=[i.replace(";"," ") for i in texts]
    texts=[i.replace("/"," ") for i in texts]
    texts=[i.replace(")"," ") for i in texts]
    texts=[i.replace("("," ") for i in texts]
    texts=[i.replace("+"," ") for i in texts]
    texts=[i.replace("="," ") for i in texts]
    texts=[i.replace("\\"," ") for i in texts]
    texts=[i.replace(":"," ") for i in texts]
    texts=[i.replace("'"," ") for i in texts]
    texts=[i.replace("`"," ") for i in texts]
    texts=[i.replace("´"," ") for i in texts]
    texts =[i.lower() for i in texts]
    #print(texts[0])
    len_texts.append(len(texts))
    print("\nNumber of patents in timeslice %i: %i\n" %(time_slice, len(texts)))
    
    del texts_raw
    """Stoplist + Formatting"""
    stoplist = ". , ! 1. 2. 3. 4. 5. 6. 7. 8. 9. 0. 1 2 3 4 5 6 7 8 9 0"
    stoplist = set(stoplist.split())
    texts=[[word.lower() for word in document.split() if not (word in stoplist or len(word)<3)] for document in texts]   
    
    #Training  
    from gensim.models import Word2Vec
    dimensions = dimensions
    window_size = window_size
    word2vec_model = Word2Vec(sentences = texts, size=dimensions, window=window_size, iter=20, min_count=min_count, workers = 1, sg=1, seed=0, hs=1, negative = noise_terms)
    word2vec_model.save("%i_W%i_T%i.word2vec_model" %(dimensions,window_size,time_slice))
    
    dictionary = {}
    similarity_value= []
    for h in bigrams:
        h = h.split(" ")
        h.sort()
        first_word, second_word = h[0], h[1]
        
        try:
             value = round(float(word2vec_model.wv.similarity(first_word, second_word)), 6)
             similarity_value.append(value)
             dictionary[str(first_word), str(second_word)] = value
        except KeyError:
            value = 0
            similarity_value.append(value)
            dictionary[str(first_word), str(second_word)] = value
             
    df = pd.DataFrame(similarity_value, index = bigrams)
    #df.to_csv("csv/word2vec_D300_W5_T%i.csv" %time_slice, sep=";", decimal=",", header=["D300_W5_T%i" %time_slice])     
    np.save("word2vec_D300_W5_T%i.npy" %time_slice, dictionary)        
    
print("\n\nProcessing time: %i seconds!\n\n" % (time() - t0))