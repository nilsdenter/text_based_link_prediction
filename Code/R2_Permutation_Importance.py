import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from time import time
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import numpy as np

np.random.seed(seed=0)

mpl.rcParams['font.size'] = 12
mpl.rcParams["font.family"] = "calibri"

"""
PERMUTATION IMPORTANCE 
"""

mpl.rcParams['font.size'] = 12
mpl.rcParams["font.family"] = "calibri"
iterations = 500
window_size_bigrams = 4
scoring = 'roc_auc'
t0 = time()

# Load file
prefix = "W"
filename = prefix + "_NE_LinkPred.csv"
non_existing_pred_data = pd.read_csv(filename, sep=";", decimal=",")
print("\nLoaded NON-existing links in {0} seconds.".format(int(time()-t0)))
t1 = time()

filename = prefix + "_E_LinkPred.csv"
existing_pred_data = pd.read_csv(filename, sep=";", decimal=",")
print("\nLoaded Existing links in {0} seconds.".format(int(time()-t1)))
df1 = pd.DataFrame(non_existing_pred_data.iloc[:,[4,5,6,7,8,9,10,11,12,13,14,15,16,17]].values)
df2 = pd.DataFrame(existing_pred_data.iloc[:,[4,5,6,7,8,9,10,11,12,13,14,15,16,17]].values)
combined_data = pd.concat([df1,df2])
rounded = combined_data.round(4)

data = combined_data.drop_duplicates(keep = "first")
data = rounded.drop_duplicates(keep = "first")
y = data.iloc[:, -1].values
X = data.iloc[:,[0,
                 1,
                 2,
                 3,
                 4,
                 5,
                 6,
                 7,
                 8,
                 9,
                 10,
                 11,
                 12
                 ]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =  0.2, random_state = 0, stratify = y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model_name = "sgd"
#load model
model = joblib.load("{0}.joblib".format(model_name))

columns = ['TF u', 'TF v', 'TFIDF u', 'TFIDF v', 'W2V cosine similarity', 'Simrank', 'RootedPageRank', 'Katz', 'CommonNeighbours', 'AdamicAdar', 'Jaccard', 'ResourceAllocation', 'PreferentialAttachment']

result_1 = permutation_importance(model, X_test, y_test, n_repeats=iterations,
                        random_state=0, n_jobs=-1, scoring="roc_auc")
sorted_idx = result_1.importances_mean.argsort()

novelty_variables = [i for i in range(0,13)]
novelty_sorted_idx = []
for entry in sorted_idx:
    if entry in novelty_variables:
        novelty_sorted_idx.append(entry)

labels = []
for entry in novelty_sorted_idx:
    labels.append(columns[entry])

fig, ax = plt.subplots(figsize=[6.4,4])
ax.boxplot(result_1.importances[novelty_sorted_idx].T,
           vert=False, labels=labels)
ax.grid()
ax.set(xlabel='\u0394 decrease in ROC AUC score', title='Permutation Importance')
ax.locator_params(tight=True, nbins=6, axis="x")

fig.tight_layout()
fig.savefig("{0}_Permutation_Technological_Importance_Test_data.png".format(model_name), dpi=1600)

#test and train beside each other
result_2 = permutation_importance(model, X_train, y_train, n_repeats=iterations,
                        random_state=0, n_jobs=-1, scoring="roc_auc")

fig, (ax1,ax2) = plt.subplots(ncols=2,nrows=1,figsize=[6.4,4], sharey=True)
mpl.rcParams['font.size'] = 10
ax1.boxplot(result_1.importances[novelty_sorted_idx].T,
           vert=False, labels=labels)
ax1.grid()
ax1.set(xlabel='\u0394 decrease in ROC AUC score')
ax1.set(title='Test Data')
ax1.locator_params(tight=True, nbins=3, axis="x")

ax2.boxplot(result_2.importances[novelty_sorted_idx].T,
           vert=False, labels=labels)
ax2.grid()
ax2.set(xlabel='\u0394 decrease in ROC AUC score')
ax2.set(title='Training Data')
ax2.locator_params(tight=True, nbins=3, axis="x")


mpl.rcParams['font.size'] = 12

fig.tight_layout()
fig.savefig("{0}_Permutation_Technological_Importance_test_train.png".format(model_name), dpi=1600)
plt.show()

df = pd.DataFrame(data=[result_1.importances_mean, result_1.importances_std, result_2.importances_mean, result_2.importances_std], columns=columns).T
df.to_excel("Permutation_Importance.xlsx", header=["Mean change in ROC AUC on test data", "SD test data", "Mean change in ROC AUC on training data", "SD training data"])