# Forecasting future bigrams and promising patents: Introducing text-based link prediction

This project contains the data and code used in the paper titled "Forecasting future bigrams and promising patents: 
Introducing text-based link prediction".

To reproduce our results, first, you need to download all files in the data repository.

Then, the code must be executed in ascending order of the sequence number.

The code "01_Word2Vec.py" creates Word2Vec embeddings of for each of the three time periods and calculates cosine similarity of all possible unigram combinations again for each time period separately.

The code "02_Generate_co_occurence_files.py" transforms a table of term frequencies of all possible unigram combinations and three periods to three files comprising a co-occurrence matrix each.

The code "03_Create_networkx_files.py" creates networkx files for each time period by using the prior created co-occurrence matrizes.

The code "04_Create_network_predictors_and_transform_data.py" calculates network-based predictors as well as the outcome variable (whether a unigram combination is mentioned in the next time period) and combines them in tabular form with predictors 1 to 5 which are directly calculated from the patent texts. In addition, the code creates four csv-files: (1) unigram combinations and their predictors of time period 1 and 2 which are mentioned in time period 3, (2) unigram combinations and their predictors of time period 1 and 2 which are not mentioned in time period 3, (3) predictors of non-existing unigram combinations of time period 3 and (4) predictors of existing unigram combinations of time period 3.

The code "05_Perform_supervised_learning.py" uses the first two files created by training and testing different classification models. In before, the data is splitted into training and test data in a 80/20 ratio. For all algorithm, 10-fold cross validation is applied and all algorithms are optimized according to a specific grid of their parameters utilizing grid search. Output files are .joblib-files of the models themselves, confusion matrizes based on training and test data as well as a tables containing statistics (e.g. ROC AUC score, true positive rates, etc.) for each classifier.

Finally, the code "06_Predict_next_time_period.py" utilizes the best performing model to construct a table of unigram combinations for each time period. In addition, the code predicts the unigram combinations in the upcoming fourth time period. The resulting table depicts all 499,500 possible unigram combinations as index and whether or not they are mentioned in the four time periods, i.e. 2005-2008, 2009-2012, 2013-2016, and the predicted 2017-2020 time period.

Note: The "STATA_Poisson_regression.txt" can be used to reproduce the statistical assessment of promising patents vs. non-promising patents.
