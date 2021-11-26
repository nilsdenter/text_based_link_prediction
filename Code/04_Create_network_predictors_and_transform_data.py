from time import time
t0 = time()
from datetime import datetime
start = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
print("\nStart: {0}".format(start))
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import pandas as pd
import networkx as nx
import linkpred as lp
import numpy as np
from linkpred.preprocess import without_selfloops

num_terms = 1000

window_size_bigrams = 4

#read word2vec
word2vec_t1 = np.load("word2vec_D300_W5_T1.npy", allow_pickle = True).item()
word2vec_t2 = np.load("word2vec_D300_W5_T2.npy", allow_pickle = True).item()
word2vec_t3 = np.load("word2vec_D300_W5_T3.npy", allow_pickle = True).item()


#read node tables
nodes_int_1 = pd.read_csv("t1_top%i.csv" %num_terms , sep= "\t", decimal = ",")
nodes_int_2 = pd.read_csv("t2_top%i.csv" %num_terms , sep= "\t", decimal = ",")
nodes_int_3 = pd.read_csv("t3_top%i.csv" %num_terms , sep= "\t", decimal = ",")

#assign term frequencies
Tf_dict_1 = {term:value for term, value in zip(nodes_int_1.Term, nodes_int_1.tf)}
Tf_dict_2 = {term:value for term, value in zip(nodes_int_2.Term, nodes_int_2.tf)}
Tf_dict_3 = {term:value for term, value in zip(nodes_int_3.Term, nodes_int_3.tf)}

#assign tfidf values
Tfidf_dict_1 = {term:value for term, value in zip(nodes_int_1.Term, nodes_int_1["tf-idf*"])}
Tfidf_dict_2 = {term:value for term, value in zip(nodes_int_2.Term, nodes_int_2["tf-idf*"])}
Tfidf_dict_3 =  {term:value for term, value in zip(nodes_int_3.Term, nodes_int_3["tf-idf*"])}

def Tuple_Sort(tupel):
    liste = list(tupel)
    liste.sort()
    return(liste[0],liste[1])
       
#load and read the first time period network file
file_name = "Term_Co_Occurence_Interval_1.gexf"
G1 = nx.read_gexf(file_name)

#load and read the second time period network file
file_name = "Term_Co_Occurence_Interval_2.gexf"
G2 = nx.read_gexf(file_name)
     
#load and read the third time period network file
file_name = "Term_Co_Occurence_Interval_3.gexf"
G3 = nx.read_gexf(file_name)
         
#transform networks to undirected
H1 = G1.to_undirected()
H1 = without_selfloops(H1)
a=0
for _ in nx.edges(H1):
    a+=1
for _ in nx.non_edges(H1):
    a+=1
print(a==((num_terms*(num_terms-1))/2))

H2 = G2.to_undirected()
H2 = without_selfloops(H2)
a=0
for _ in nx.edges(H2):
    a+=1
for _ in nx.non_edges(H2):
    a+=1
print(a==((num_terms*(num_terms-1))/2))

H3 = G3.to_undirected()
H3 = without_selfloops(H3)
a=0
for _ in nx.edges(H3):
    a+=1
for _ in nx.non_edges(H3):
    a+=1
print(a==((num_terms*(num_terms-1))/2))

#Set weight 
weight = None
name_prefix = "W"

result_list = []
i = 0 
interval = 1
for network, nextnetwork, word2vec, tf_dict, tfidf_dict in [(H1,H2,word2vec_t1, Tf_dict_1, Tfidf_dict_1), (H2, H3, word2vec_t2, Tf_dict_2, Tfidf_dict_2)]:
    #, (H2,H3), (H3, H4)]:
    #Existing Edges
    
    print("\nCalculating Preferential Attachment")
    pad_results = {}
    pa_preds = nx.preferential_attachment(network, nx.edges(network))
    for u, v, p, in pa_preds:
        edge = (u,v)
        pad_results[Tuple_Sort(edge)] = p
    
    print("\nCalculating Simrank")
    simrank = lp.predictors.SimRank(network, excluded = nx.non_edges(network))
    simrank_results = simrank.predict(c=0.85, weight = weight)
    
    print("\nCalculating RootedPageRank")
    rootedpagerank = lp.predictors.RootedPageRank(network, excluded = nx.non_edges(network))
    rpr_results = rootedpagerank.predict(weight = weight)
    
    print("\nCalculating Katz")
    katz = lp.predictors.Katz(network, excluded = nx.non_edges(network))
    katz_results = katz.predict(weight = weight)
    
    print("\nCalculating CommonNeighbours")
    commonneighbours = lp.predictors.CommonNeighbours(network, excluded = nx.non_edges(network))
    cn_results = commonneighbours.predict(weight = weight)
    
    print("\nCalculating AdamicAdar")
    adamicadar = lp.predictors.AdamicAdar(network, excluded = nx.non_edges(network))
    adamicadar_results = adamicadar.predict(weight = weight)
    
    print("\nCalculating Jaccard")
    jaccard = lp.predictors.Jaccard(network, excluded = nx.non_edges(network))
    jaccard_results = jaccard.predict(weight = weight)
    
    print("\nCalculating ResourceAllocation")
    resourceallocation = lp.predictors.ResourceAllocation(network, excluded = nx.non_edges(network))
    ra_results = resourceallocation.predict(weight = weight)
    
    print("\nStarting DataFrame assembly")
    a=0
    for edge in nx.edges(network):
        edge = Tuple_Sort(edge)
        word_similarity = word2vec[edge]
        tf_value_1 = tf_dict[edge[0]]
        tf_value_2 = tf_dict[edge[1]]
        tfidf_value_1 = tfidf_dict[edge[0]]
        tfidf_value_2 = tfidf_dict[edge[1]]
        result_list.append([interval, edge[0], edge[1], tf_value_1, tf_value_2, tfidf_value_1, tfidf_value_2, word_similarity, simrank_results[edge], rpr_results[edge], katz_results[edge], cn_results[edge], adamicadar_results[edge], jaccard_results[edge], ra_results[edge], pad_results[edge], nextnetwork.has_edge(edge[0],edge[1])])
        a += 1
        if a % 10000 == 0:
            print(a)
    
    print("\nInterval %i of 2 of list 1 of 4 completed" %(interval))
    interval += 1
    
existing_pred_data = pd.DataFrame(result_list, columns = ["NetworkInterval","u","v", "TF_u","TF_v","TFIDF_u","TFIDF_v","Word_Similarity", "Simrank","RootedPageRank","Katz","CommonNeighbours","AdamicAdar", "Jaccard","ResourceAllocation","PreferentialAttachment","Existing_in_T+1"])
existing_pred_data["Existing_in_T+1"] = [int(x) for x in existing_pred_data["Existing_in_T+1"]]

#Save Files
save_name = name_prefix + "_E_LinkPred.csv"
existing_pred_data.to_csv(save_name, sep=";", decimal=",")
print("\n1 of 4 finished\n")

result_list = []
interval = 1
for network, nextnetwork, word2vec, tf_dict, tfidf_dict in [(H1,H2,word2vec_t1, Tf_dict_1, Tfidf_dict_1), (H2,H3,word2vec_t2,Tf_dict_2,Tfidf_dict_2)]:  
    #Non-Existing Edges
    print("\nCalculating Preferential Attachment")
    pad_results = {}
    pa_preds = nx.preferential_attachment(network, nx.non_edges(network))
    for u, v, p, in pa_preds:
        edge = (u,v)
        pad_results[Tuple_Sort(edge)] = p
    
    print("\nCalculating Simrank")
    simrank = lp.predictors.SimRank(network, excluded = nx.edges(network))
    simrank_results = simrank.predict(c=0.85, weight = weight)
    
    print("\nCalculating RootedPageRank")
    rootedpagerank = lp.predictors.RootedPageRank(network, excluded = nx.edges(network))
    rpr_results = rootedpagerank.predict(weight = weight)
    
    print("\nCalculating Katz")
    katz = lp.predictors.Katz(network, excluded = nx.edges(network))
    katz_results = katz.predict(weight = weight)
    
    print("Calculating CommonNeighbours")
    commonneighbours = lp.predictors.CommonNeighbours(network, excluded = nx.edges(network))
    cn_results = commonneighbours.predict(weight = weight)
    
    print("\nCalculating AdamicAdar")
    adamicadar = lp.predictors.AdamicAdar(network, excluded = nx.edges(network))
    adamicadar_results = adamicadar.predict(weight = weight)
    
    print("\nCalculating Jaccard")
    jaccard = lp.predictors.Jaccard(network, excluded = nx.edges(network))
    jaccard_results = jaccard.predict(weight = weight)
    
    print("\nCalculating ResourceAllocation")
    resourceallocation = lp.predictors.ResourceAllocation(network, excluded = nx.edges(network))
    ra_results = resourceallocation.predict(weight = weight)
    
    print("\nStarting DataFrame assembly")
    a = 1
    for edge in nx.non_edges(network):
        edge = Tuple_Sort(edge)
        word_similarity = word2vec[edge]
        tf_value_1 = tf_dict[edge[0]]
        tf_value_2 = tf_dict[edge[1]]
        tfidf_value_1 = tfidf_dict[edge[0]]
        tfidf_value_2 = tfidf_dict[edge[1]]
        result_list.append([interval, edge[0], edge[1], tf_value_1, tf_value_2, tfidf_value_1, tfidf_value_2, word_similarity, simrank_results[edge], rpr_results[edge], katz_results[edge], cn_results[edge], adamicadar_results[edge], jaccard_results[edge], ra_results[edge], pad_results[edge], nextnetwork.has_edge(edge[0],edge[1])])
        a += 1
        if a % 10000 == 0:
            print(a)
    print("\nInterval %i of 2 of list 2 of 4 completed" %(interval))
    interval += 1
    
non_existing_pred_data = pd.DataFrame(result_list, columns = ["NetworkInterval","u","v", "TF_u","TF_v","TFIDF_u","TFIDF_v","Word_Similarity", "Simrank","RootedPageRank","Katz","CommonNeighbours","AdamicAdar", "Jaccard","ResourceAllocation","PreferentialAttachment","Existing_in_T+1"])
non_existing_pred_data["Existing_in_T+1"] = [int(x) for x in non_existing_pred_data["Existing_in_T+1"]]

#Save Files
save_name = name_prefix + "_NE_LinkPred.csv"
non_existing_pred_data.to_csv(save_name, sep=";", decimal=",")
print("\n2 of 4 finished\n")

###############################################################################

#Handle Interval 3
interval=3
#Initiate Dataframe(s)
result_list = []
tf_dict = Tf_dict_3
tfidf_dict = Tfidf_dict_3 
network = H3
    ##Existing Edges
print("\nCalculating Simrank")
simrank = lp.predictors.SimRank(network, excluded = nx.non_edges(network))
simrank_results = simrank.predict(c=0.85, weight = weight)

print("\nCalculating RootedPageRank")
rootedpagerank = lp.predictors.RootedPageRank(network, excluded = nx.non_edges(network))
rpr_results = rootedpagerank.predict(weight = weight)

print("\nCalculating Katz")
katz = lp.predictors.Katz(network, excluded = nx.non_edges(network))
katz_results = katz.predict(weight = weight)

print("\nCalculating CommonNeighbours")
commonneighbours = lp.predictors.CommonNeighbours(network, excluded = nx.non_edges(network))
cn_results = commonneighbours.predict(weight = weight)

print("\nCalculating AdamicAdar")
adamicadar = lp.predictors.AdamicAdar(network, excluded = nx.non_edges(network))
adamicadar_results = adamicadar.predict(weight = weight)

print("\nCalculating Jaccard")
jaccard = lp.predictors.Jaccard(network, excluded = nx.non_edges(network))
jaccard_results = jaccard.predict(weight = weight)

print("\nCalculating ResourceAllocation")
resourceallocation = lp.predictors.ResourceAllocation(network, excluded = nx.non_edges(network))
ra_results = resourceallocation.predict(weight = weight)

pad_results = {}
pa_preds = nx.preferential_attachment(network, nx.edges(network))
for u, v, p, in pa_preds:
    edge = (u,v)
    pad_results[Tuple_Sort(edge)] = p


for edge in nx.edges(network):
    edge = Tuple_Sort(edge)
    word_similarity = word2vec[edge]
    tf_value_1 = tf_dict[edge[0]]
    tf_value_2 = tf_dict[edge[1]]
    tfidf_value_1 = tfidf_dict[edge[0]]
    tfidf_value_2 = tfidf_dict[edge[1]]
    result_list.append([interval, edge[0], edge[1], tf_value_1, tf_value_2, tfidf_value_1, tfidf_value_2, word_similarity, simrank_results[edge], rpr_results[edge], katz_results[edge], cn_results[edge], adamicadar_results[edge], jaccard_results[edge], ra_results[edge], pad_results[edge]])
    a += 1
    if a % 10000 == 0:
        print(a)
    
existing_i3_pred_data = pd.DataFrame(result_list, columns = ["NetworkInterval","u","v", "TF_u","TF_v","TFIDF_u","TFIDF_v","Word_Similarity", "Simrank","RootedPageRank","Katz","CommonNeighbours","AdamicAdar", "Jaccard","ResourceAllocation","PreferentialAttachment"])

#Save Files
save_name = name_prefix + "_LinkPred_i3_ExistingLinks.csv"
existing_i3_pred_data.to_csv(save_name, sep=";", decimal=",")
print("\n3 of 4 finished\n")

#Initiate Dataframe(s)
i = 0 
network = H3
result_list = []
interval=3
## Non-Existing Edges
print("\nCalculating Simrank")
simrank = lp.predictors.SimRank(network, excluded = nx.edges(network))
simrank_results = simrank.predict(c=0.85, weight = weight)

print("\nCalculating RootedPageRank")
rootedpagerank = lp.predictors.RootedPageRank(network, excluded = nx.edges(network))
rpr_results = rootedpagerank.predict(weight = weight)

print("\nCalculating Katz")
katz = lp.predictors.Katz(network, excluded = nx.non_edges(network))
katz_results = katz.predict(weight = weight)

print("\nCalculating CommonNeighbours")
commonneighbours = lp.predictors.CommonNeighbours(network, excluded = nx.edges(network))
cn_results = commonneighbours.predict(weight = weight)

print("\nCalculating AdamicAdar")
adamicadar = lp.predictors.AdamicAdar(network, excluded = nx.edges(network))
adamicadar_results = adamicadar.predict(weight = weight)

print("\nCalculating Jaccard")
jaccard = lp.predictors.Jaccard(network, excluded = nx.edges(network))
jaccard_results = jaccard.predict(weight = weight)

print("\nCalculating ResourceAllocation")
resourceallocation = lp.predictors.ResourceAllocation(network, excluded = nx.edges(network))
ra_results = resourceallocation.predict(weight = weight)
    
print("\nCalculating Preferential Attachment")
pad_results = {}
pa_preds = nx.preferential_attachment(network, nx.non_edges(network))
for u, v, p, in pa_preds:
    edge = (u,v)
    pad_results[Tuple_Sort(edge)] = p

print("\nStarting DataFrame assembly")
for edge in nx.non_edges(network):
    edge = Tuple_Sort(edge)
    word_similarity = word2vec[edge]
    tf_value_1 = tf_dict[edge[0]]
    tf_value_2 = tf_dict[edge[1]]
    tfidf_value_1 = tfidf_dict[edge[0]]
    tfidf_value_2 = tfidf_dict[edge[1]]
    result_list.append([interval, edge[0], edge[1], tf_value_1, tf_value_2, tfidf_value_1, tfidf_value_2, word_similarity, simrank_results[edge], rpr_results[edge], katz_results[edge], cn_results[edge], adamicadar_results[edge], jaccard_results[edge], ra_results[edge], pad_results[edge]])
    a += 1
    if a % 10000 == 0:
        print(a)


non_existing_i3_pred_data = pd.DataFrame(result_list, columns = ["NetworkInterval","u","v", "TF_u","TF_v","TFIDF_u","TFIDF_v","Word_Similarity", "Simrank","RootedPageRank","Katz","CommonNeighbours","AdamicAdar", "Jaccard","ResourceAllocation","PreferentialAttachment"])

#Save Files
save_name = name_prefix + "_LinkPred_i3_NonExistingLinks.csv"
non_existing_i3_pred_data.to_csv(save_name, sep=";", decimal=",")

print("\nStarted at {0}, finished at {1}, passed time: {2} seconds.\n".format(start, datetime.now().strftime("%d-%m-%Y %H:%M:%S"), int(time()-t0)))