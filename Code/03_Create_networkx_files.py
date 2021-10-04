import pandas as pd
from time import time
t0 = time()
from datetime import datetime

start = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
print("\nStart: {0}\n".format(start))

window_size_bigrams = 4

#Read link tables
co_occurance_t1 = pd.read_csv("wl4_n2_w%i_t1_4years_tf_cooccurence.csv" %window_size_bigrams, sep= ";", decimal = ",")
co_occurance_t2 = pd.read_csv("wl4_n2_w%i_t2_4years_tf_cooccurence.csv" %window_size_bigrams, sep= ";", decimal = ",")
co_occurance_t3 = pd.read_csv("wl4_n2_w%i_t3_4years_tf_cooccurence.csv" %window_size_bigrams, sep= ";", decimal = ",")

print("Time to read data: {0} seconds\n".format(int(time()-t0)))


# =============================================================================
# Create Networks
# =============================================================================
#Clean Data
def clean_Intervall_Lists(table):
    table = table.set_index("Unnamed: 0", drop = True)
    return table

clean_co_occurance_t1 = clean_Intervall_Lists(co_occurance_t1)
clean_co_occurance_t2 = clean_Intervall_Lists(co_occurance_t2)
clean_co_occurance_t3 = clean_Intervall_Lists(co_occurance_t3)

term_list = list(co_occurance_t1["Unnamed: 0"])
def sort_Tuples(input_tuple):
    tuple_list = list(input_tuple)
    tuple_list.sort()
    return tuple(tuple_list)
    
from itertools import combinations

combination_dict_t1 = {}
combination_dict_t2 = {}
combination_dict_t3 = {}

for combination_dict, co_occurance_matrix in zip([combination_dict_t1,combination_dict_t2,combination_dict_t3],[clean_co_occurance_t1,clean_co_occurance_t2,clean_co_occurance_t3]):
    for combination in combinations(term_list,2):

        new_combination = sort_Tuples(combination)
        value = co_occurance_matrix.loc[new_combination[0],new_combination[1]]
        if str(value) == "nan":
            continue
        else: 
            combination_dict[new_combination] = value
import networkx as nx
for i, combination_dict in zip([1,2,3],[combination_dict_t1,combination_dict_t2,combination_dict_t3]):
    G = nx.Graph()
    for key in combination_dict.keys():
        G.add_edge(key[0], key[1], weight = combination_dict[key])
        
    save_file = "Term_Co_Occurence_Interval_" + str(i) + ".gexf"
    nx.write_gexf(G,save_file)
    
print("Started at {0}, finished at {1}, passed time: {2} seconds.\n".format(start, datetime.now().strftime("%d-%m-%Y %H:%M:%S"), int(time()-t0)))