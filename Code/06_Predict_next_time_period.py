from time import time
from datetime import datetime
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import pandas as pd




#model parameters
best_model = "sgd"

t0 = time()
start = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
print("\nStart: {0}".format(start))

# Load interval 3
filename = "W_LinkPred_i3_NonExistingLinks.csv"
non_existing_links_i3 = pd.read_csv(filename, sep=";", decimal=",")
print("\nLoaded NON-existing links of interval 3")

filename = "W_LinkPred_i3_ExistingLinks.csv"
existing_links_i3 = pd.read_csv(filename, sep=";", decimal=",")
print("\nLoaded Existing links of interval 3")

non_existing_links_i3["i3"]= [0 for i in range(len(non_existing_links_i3))]
existing_links_i3["i3"]= [1 for i in range(len(existing_links_i3))]
data_i3 = pd.concat([non_existing_links_i3,existing_links_i3])
data_i3 = data_i3.sort_values(by=["u","v"])
print("\n"+str(len(data_i3)))

X_predict = pd.DataFrame(data_i3.iloc[:,[4,5,6,7,8,9,10,11,12,13,14,15,16]].values)
data = X_predict.round(4)
X = data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12]].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_predict = sc.fit_transform(X)


# Load intervals 1 to 4
filename = "W_NE_LinkPred.csv"
non_existing_links_other_i = pd.read_csv(filename, sep=";", decimal=",")
print("\nLoaded NON-existing links of interval 1 to 4.")
t1 = time()

filename = "W_E_LinkPred.csv"
existing_links_other_i = pd.read_csv(filename, sep=";", decimal=",")
print("\nLoaded Existing links of interval 1 to 4")

# i1
existing_links_i1 = existing_links_other_i.loc[existing_links_other_i['NetworkInterval']==1]
existing_links_i1["i1"]= [1 for i in range(len(existing_links_i1))]
non_existing_links_i1 = non_existing_links_other_i.loc[non_existing_links_other_i['NetworkInterval']==1]
non_existing_links_i1["i1"]= [0 for i in range(len(non_existing_links_i1))]
combined_data_i1 = pd.concat([non_existing_links_i1,existing_links_i1])
combined_data_i1 = combined_data_i1.sort_values(by=["u","v"])
data_i1 = pd.DataFrame(combined_data_i1.iloc[:,[2,3,-1]].values, columns=["u","v","i1"])
print("\n"+str(len(data_i1)))
#data_i1.to_excel("final_table_i1.xlsx")

# i2
existing_links_i2 = existing_links_other_i.loc[existing_links_other_i['NetworkInterval']==2]
existing_links_i2["i2"]= [1 for i in range(len(existing_links_i2))]
non_existing_links_i2 = non_existing_links_other_i.loc[non_existing_links_other_i['NetworkInterval']==2]
non_existing_links_i2["i2"]= [0 for i in range(len(non_existing_links_i2))]
combined_data_i2 = pd.concat([non_existing_links_i2,existing_links_i2])
combined_data_i2 = combined_data_i2.sort_values(by=["u","v"])
data_i2 = pd.DataFrame(combined_data_i2.iloc[:,[2,3,-1]].values, columns=["u","v","i2"])
print("\n"+str(len(data_i2)))
#data_i2.to_excel("final_table_i2.xlsx")

data_i3 = pd.DataFrame(data_i3.iloc[:,[2,3,-1]].values, columns=["u","v","i3"])
#data_i3.to_excel("final_table_i3.xlsx")

final_table = pd.DataFrame()
final_table["u"] = data_i1["u"]
final_table["v"] = data_i1["v"]
final_table["2005-2008"] = data_i1["i1"]
final_table["2009-2012"] = data_i2["i2"]
final_table["2013-2016"] = data_i3["i3"]

# Predict model
import joblib
model = joblib.load("%s.joblib" %best_model)
print("\nStarted prediction at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))

# predict interval 4
model_pred = model.predict(X_predict)

final_table["2017-2020"] = model_pred

print(final_table.head())
final_table.to_excel("%s_Final_Table_1-4.xlsx" %(best_model), index=False)