from time import time
from datetime import datetime
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, confusion_matrix, roc_auc_score, precision_score,f1_score, recall_score, roc_curve

def analytics(y_pred, classifier):
    fpr, tpr, threshold = roc_curve(y_true = y_test, y_score = y_pred, pos_label = 1)
    print("\n")
    return fpr, tpr, threshold

from collections import defaultdict
statistics = defaultdict(list)

def generate_statistics(classifier, method, model_pred_train, model_pred_test):
    statistics["Balanced accuracy on test data"].append(round(balanced_accuracy_score(y_test, model_pred_test),4))
    statistics["ROC AUC score on test data"].append(round(roc_auc_score(y_test, model_pred_test),4))
    statistics["Positive predictive value on test data"].append(round(precision_score(y_test, model_pred_test, pos_label=1),4))
    statistics["True positive rate on test data"].append(round(recall_score(y_test, model_pred_test, pos_label=1),4))
    statistics["Negative predictive value on test data"].append(round(precision_score(y_test, model_pred_test, pos_label=0),4))
    statistics["True negative rate on test data"].append(round(recall_score(y_test, model_pred_test, pos_label=0),4))
    statistics["F1 score on test data"].append(round(f1_score(y_test, model_pred_test),4))
    statistics["MCC on test data"].append(round(matthews_corrcoef(y_test, model_pred_test),4))
    statistics["Balanced accuracy on train data"].append(round(balanced_accuracy_score(y_train, model_pred_train),4))
    statistics["ROC AUC score on train data"].append(round(roc_auc_score(y_train, model_pred_train),4))
    statistics["Positive predictive value on train data"].append(round(precision_score(y_train, model_pred_train, pos_label=1),4))
    statistics["True positive rate on train data"].append(round(recall_score(y_train, model_pred_train, pos_label=1),4))
    statistics["Negative predictive value on train data"].append(round(precision_score(y_train, model_pred_train, pos_label=0),4))
    statistics["True negative rate on train data"].append(round(recall_score(y_train, model_pred_train, pos_label=0),4))
    statistics["F1 score on train data"].append(round(f1_score(y_train, model_pred_train),4))
    statistics["MCC on train data"].append(round(matthews_corrcoef(y_train, model_pred_train),4))
    model_confusion_matrix = confusion_matrix(y_test, model_pred_test, labels = [1,0])
    model_confusion_matrix = pd.DataFrame(model_confusion_matrix, index= ["Actual class: Presence", "Actual class: Absence"], columns = ["Predicted class: Presence", "Predicted class: Absence"] )
    model_confusion_matrix.to_excel("%s_confusion_matrix_on_test_data.xlsx" %method)
    print(model_confusion_matrix)
    model_confusion_matrix = confusion_matrix(y_train, model_pred_train, labels = [1,0])
    model_confusion_matrix = pd.DataFrame(model_confusion_matrix, index= ["Actual class: Presence", "Actual class: Absence"], columns = ["Predicted class: Presence", "Predicted class: Absence"] )
    model_confusion_matrix.to_excel("%s_confusion_matrix_on_train_data.xlsx" %method)
    print(model_confusion_matrix)
    try: 
        print("\nBest model by grid search:")
        print(classifier.best_estimator_)
        print("\nBest Score by grid search:")
        print(classifier.best_score_)
        statistics["Parameter"].append(str(classifier.best_estimator_))
        pd.DataFrame(classifier.cv_results_).to_excel("%s_parameters.xlsx" %method, index=False)
        statistics["ROC AUC score mean on validation data"].append(round(classifier.best_score_,4))
    except:
        print("Dummy!")
        statistics["Parameter"].append("Dummy")
        statistics["ROC AUC score mean on validation data"].append("-")

start = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
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

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 0, stratify = y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

statistics["Method"].append("StochasticGradientDescent")
statistics["Train sample size"].append(len(y_train))
statistics["Test sample size"].append(len(y_test))
print("\nStarted fitting Stochastic Gradient Descent Classifier {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
t1 = time()

params_sgd = {
    "loss" : ["hinge", "log", "huber", "modified_huber", "perceptron", "epsilon_insensitive"],
    "alpha" : [10**-i for i in range (1, 6)],
    "penalty" : ["l2", "elasticnet"],
    "class_weight" : [None],
    "tol": [0.001],
    "max_iter": [1000],
    "l1_ratio": [i/10 for i in range(1,10)]
}

model = SGDClassifier(random_state = 0, shuffle=True)
sgd = GridSearchCV(estimator = model, 
                           param_grid = params_sgd,
                           scoring = 'roc_auc',
                           cv = 10,
                           n_jobs = -1,
                           return_train_score=True,
                           verbose=10)
sgd.fit(X_train, y_train)
statistics["Training Duration"].append(int(time()-t1))
joblib.dump(sgd, "sgd.joblib")
t1 = time()
sgd_pred = sgd.predict(X_test)
sgd_pred_train = sgd.predict(X_train)
statistics["Prediction Duration"].append(int(time()-t1))
print("\nAnalytics Stochastic Gradient Descent Classifier at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
sgd_fpr, sgd_tpr, sgd_threshold = analytics(sgd_pred, sgd)
generate_statistics(classifier=sgd, method="sgd", model_pred_train=sgd_pred_train, model_pred_test=sgd_pred)
del sgd


statistics["Method"].append("GaussianNaiveBayes")
statistics["Train sample size"].append(len(y_train))
statistics["Test sample size"].append(len(y_test))
print("\nStarted fitting GausianNaiveBayes at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
t1 = time()
params_gnb = {
    "var_smoothing" : [10000/(10**i) for i in range(0, 21)]
}
model = GaussianNB()
gnb = GridSearchCV(estimator = model, 
                           param_grid = params_gnb,
                           scoring = 'roc_auc',
                           cv = 10,
                           n_jobs = -1,
                           return_train_score=True,
                           verbose=10)
gnb.fit(X_train, y_train)
statistics["Training Duration"].append(int(time()-t1))
joblib.dump(gnb, "gnb.joblib")
t1 = time()
gnb_pred = gnb.predict(X_test)
gnb_pred_train = gnb.predict(X_train)
statistics["Prediction Duration"].append(int(time()-t1))
print("\nAnalytics GausianNaiveBayes at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
gnb_fpr, gnb_tpr, gnb_threshold = analytics(gnb_pred, gnb)
generate_statistics(classifier=gnb, method="gnb", model_pred_train=gnb_pred_train, model_pred_test=gnb_pred)
del gnb


statistics["Method"].append("DecisionTreeClassifier")
statistics["Train sample size"].append(len(y_train))
statistics["Test sample size"].append(len(y_test))
print("\nStarted fitting DecisionTreeClassifier at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
t1 = time()
params_dtc = {
    "criterion" : ["gini", "entropy"],
    "splitter": ["best", "random"],
    "class_weight": [None],
    "max_depth": [2,3,5,7,10,15,20,25,30],
    "max_leaf_nodes" : [i for i in range (2,12)]
}
model = DecisionTreeClassifier(random_state = 0)
dtc = GridSearchCV(estimator = model, 
                           param_grid = params_dtc,
                           scoring = 'roc_auc',
                           cv = 10,
                           n_jobs = -1,
                           return_train_score=True,
                           verbose=10)
dtc.fit(X_train, y_train)
statistics["Training Duration"].append(int(time()-t1))
joblib.dump(dtc, "dtc.joblib")
t1 = time()
dtc_pred = dtc.predict(X_test)
dtc_pred_train = dtc.predict(X_train)
statistics["Prediction Duration"].append(int(time()-t1))
print("\nAnalytics DecisionTreeClassifier at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
dtc_fpr, dtc_tpr, dtc_threshold = analytics(dtc_pred, dtc)
generate_statistics(classifier=dtc, method="dtc", model_pred_train=dtc_pred_train, model_pred_test=dtc_pred)
del dtc


statistics["Method"].append("RandomForestClassifier")
statistics["Train sample size"].append(len(y_train))
statistics["Test sample size"].append(len(y_test))
print("\nStarted fitting RandomForestClassifier at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
t1 = time()
params_rfc = {
    "criterion" : ["gini", "entropy"],
    "n_estimators": [15,20,30,40,50,75,100,150],
    "max_depth": [1,2,3,5,10,20,30,50],
    'max_features': ['auto', 'log2', 'sqrt'],
    "max_leaf_nodes" : [i for i in range (2,12)]
}

model = RandomForestClassifier(random_state = 0)
rfc = GridSearchCV(estimator = model, 
                           param_grid = params_rfc,
                           scoring = 'roc_auc',
                           cv = 10,
                           n_jobs = -1,
                           return_train_score=True,
                           verbose=10)
rfc.fit(X_train, y_train)
statistics["Training Duration"].append(int(time()-t1))
joblib.dump(rfc, "rfc.joblib")
t1 = time()
rfc_pred = rfc.predict(X_test)
rfc_pred_train = rfc.predict(X_train)
statistics["Prediction Duration"].append(int(time()-t1))
print("\nAnalytics RandomForestClassifier at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
rfc_fpr, rfc_tpr, rfc_threshold = analytics(rfc_pred, rfc)
generate_statistics(classifier=rfc, method="rfc", model_pred_train=rfc_pred_train, model_pred_test=rfc_pred)
del rfc


statistics["Method"].append("KNeighborsClassifier")
statistics["Train sample size"].append(len(y_train))
statistics["Test sample size"].append(len(y_test))
print("\nStarted fitting KNeighborsClassifier at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
t1 = time()
from sklearn.neighbors import KNeighborsClassifier
params_knn = {
    "metric" : ["euclidean", "manhattan", "minkowski"],
    'n_neighbors' : [5, 10, 15, 20, 25, 30]
    }
model = KNeighborsClassifier()
knn = GridSearchCV(estimator = model, 
                           param_grid = params_knn,
                           scoring = 'roc_auc',
                           cv = 10,
                           n_jobs = -1,
                           return_train_score=True,
                           verbose=10)
knn.fit(X_train, y_train)
statistics["Training Duration"].append(int(time()-t1))
joblib.dump(knn, "knn.joblib")
t1 = time()
knn_pred = knn.predict(X_test)
knn_pred_train = knn.predict(X_train)
statistics["Prediction Duration"].append(int(time()-t1))
print("\nAnalytics KNN at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
knn_fpr, knn_tpr, knn_threshold = analytics(knn_pred, knn)
generate_statistics(classifier=knn, method="knn", model_pred_train=knn_pred_train, model_pred_test=knn_pred)
del knn


print("\nStarted fitting LogisticRegression at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
statistics["Method"].append("LogisticRegression")
statistics["Train sample size"].append(len(y_train))
statistics["Test sample size"].append(len(y_test))
t1 = time()
params_logistic = {
    "penalty" : ["l1", "l2", "elasticnet"],
    "C" : [100,10,1,0.5, 0.1, 0.05, 0.01],
    "solver" : ["saga"]}

model = LogisticRegression(random_state = 0)
logistic = GridSearchCV(estimator = model, 
                           param_grid = params_logistic,
                           scoring = 'roc_auc',
                           cv = 10,
                           n_jobs = -1,
                           return_train_score=True,
                           verbose=10)
logistic.fit(X_train, y_train)
statistics["Training Duration"].append(int(time()-t1))
joblib.dump(logistic, "logistic.joblib")
t1 = time()
logistic_pred = logistic.predict(X_test)
logistic_pred_train = logistic.predict(X_train)
statistics["Prediction Duration"].append(int(time()-t1))
print("\nAnalytics Logistic Regression at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
logistic_fpr, logistic_tpr, logistic_threshold = analytics(logistic_pred, logistic)
generate_statistics(classifier=logistic, method="logistic", model_pred_train=logistic_pred_train, model_pred_test=logistic_pred)
del logistic


statistics["Method"].append("DummyClassifier")
statistics["Train sample size"].append(len(y_train))
statistics["Test sample size"].append(len(y_test))
print("\nStarted fitting DummyClassifier at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
t1 = time()
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(random_state = 0)
dummy.fit(X_train, y_train)
statistics["Training Duration"].append(int(time()-t1))
joblib.dump(dummy, "dummy.joblib")
t1 = time()
dummy_pred = dummy.predict(X_test)
dummy_pred_train = dummy.predict(X_train)
statistics["Prediction Duration"].append(int(time()-t1))
print("\nAnalytics Dummy Classifier at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
dummy_fpr, dummy_tpr, dummy_threshold = analytics(dummy_pred, dummy)
generate_statistics(classifier=dummy, method="dummy", model_pred_train=dummy_pred_train, model_pred_test=dummy_pred)
del dummy



print("\nStarted fitting Voting Classifier at {0}".format(datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
t1 = time()
statistics["Method"].append("Voting Classifier")
statistics["Train sample size"].append(len(y_train))
statistics["Test sample size"].append(len(y_test))

#Voting classifier of best models
best_sgd = SGDClassifier(random_state = 0, shuffle=True, 
    loss = "log",
    alpha = 0.001,
    penalty = "elasticnet",
    class_weight = None,
    tol= 0.001,
    max_iter= 1000,
    l1_ratio= 0.9)
best_gnb = GaussianNB(
    var_smoothing= 1)
best_dtc = DecisionTreeClassifier(random_state = 0,    
    criterion = "entropy",
    splitter= "best",
    class_weight= None,
    max_depth= 10,
    max_leaf_nodes = 10)
best_rfc = RandomForestClassifier(random_state = 0, 
    criterion = "entropy",
    n_estimators= 30,
    max_depth= 10,
    max_features= 'auto',
    max_leaf_nodes = 10)
best_logistic = LogisticRegression(random_state=0, 
                                   penalty="l1",
                                   solver="saga")
best_knn = KNeighborsClassifier(metric="euclidean", n_neighbors=10)

classifiers = [('sgd', best_sgd), ('gnb', best_gnb), ('dtc', best_dtc), ('rfc', best_rfc), ('logistic', best_logistic), ('knn', best_knn)]

pipe = Pipeline([
    ('votingclassifier', VotingClassifier(
        voting='soft',
        estimators=classifiers))])
                       
def combinations_on_off(num_classifiers):
    return [[int(x) for x in list("{0:0b}".format(i).zfill(num_classifiers))]
            for i in range(1, 2 ** num_classifiers)]

combinations = []
for i in combinations_on_off(len(classifiers)):
    if sum(i)>1:
        combinations.append(i)

param_votingclassifier = dict(
    votingclassifier__weights=combinations)

voting = GridSearchCV(estimator = pipe, 
                           param_grid = param_votingclassifier,
                           scoring = 'roc_auc',
                           cv = 10,
                           n_jobs = -1,
                           return_train_score=True,
                           verbose=10)
voting.fit(X_train, y_train)
statistics["Training Duration"].append(int(time()-t1))
joblib.dump(voting, "voting.joblib")
t1 = time()
voting_pred = voting.predict(X_test)
voting_pred_train = voting.predict(X_train)
statistics["Prediction Duration"].append(int(time()-t1))
voting_fpr, voting_tpr, voting_threshold = analytics(voting_pred, voting)
generate_statistics(classifier=voting, method="voting", model_pred_train=voting_pred_train, model_pred_test=voting_pred)
del voting

df3 = pd.DataFrame(data = statistics)
df3 = df3.sort_values(by=["ROC AUC score on test data"], ascending=False)
df3.to_excel("statistics.xlsx", index=False)
print("\nStarted at {0}, finished at {1}, passed time: {2} seconds.\n".format(start, datetime.now().strftime("%d-%m-%Y %H:%M:%S"), int(time()-t0)))