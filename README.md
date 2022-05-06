# Forecasting future bigrams and promising patents: Introducing text-based link prediction

This project contains the data and code used in the paper: Denter, Nils M.; Aaldering, Lukas Jan; Caferoglu, Huseyin (2022): Forecasting future bigrams and promising patents: Introducing text-based link prediction. In Foresight ahead-of-print (ahead-of-print). DOI: doi.org/10.1108/fs-03-2021-0078.

To reproduce our results, first, you need to download all files in the data folder.

Then, the code must be executed in ascending order of the sequence number.

The code "01_Word2Vec.py" creates Word2Vec embeddings of for each of the three time periods and calculates cosine similarity of all possible unigram combinations again for each time period separately.

The code "02_Generate_co_occurence_files.py" transforms a table of term frequencies of all possible unigram combinations and three periods to three files comprising a co-occurrence matrix each.

The code "03_Create_networkx_files.py" creates networkx files for each time period by using the prior created co-occurrence matrizes.

The code "04_Create_network_predictors_and_transform_data.py" calculates network-based predictors as well as the outcome variable (whether a unigram combination is mentioned in the next time period) and combines them in tabular form with predictors 1 to 5 which are directly calculated from the patent texts. In addition, the code creates four csv-files: (1) unigram combinations and their predictors of time period 1 and 2 which are mentioned in time period 3, (2) unigram combinations and their predictors of time period 1 and 2 which are not mentioned in time period 3, (3) predictors of non-existing unigram combinations of time period 3 and (4) predictors of existing unigram combinations of time period 3.

The code "05_Perform_supervised_learning.py" uses the first two files created by training and testing different classification models. In before, the data is splitted into training and test data in a 80/20 ratio. For all algorithm, 10-fold cross validation is applied and all algorithms are optimized according to a specific grid of their parameters utilizing grid search. Output files are .joblib-files of the models themselves, confusion matrizes based on training and test data as well as a tables containing statistics (e.g. ROC AUC score, true positive rates, etc.) for each classifier.

The code "06_Predict_next_time_period.py" utilizes the best performing model to construct a table of unigram combinations for each time period. In addition, the code predicts the unigram combinations in the upcoming fourth time period. The resulting table depicts all 499,500 possible unigram combinations as index and whether or not they are mentioned in the four time periods, i.e. 2005-2008, 2009-2012, 2013-2016, and the predicted 2017-2020 time period.

Finally, the code "07_Future_Bigrams_and_Promising_Patents.py" uses a list of unigram combinations that are predicted to be mentioned in time period 4 but are not mentioned yet in the time periods before. We refer to these unigram combinations are future bigrams. For each future bigram, the code checks whether the bigram's unigrams co-occur in the patents of the time period 3 and obtains a "promisingness" score for each patent as the number of different future bigrams whose unigrams co-occur within the patent. Consequently, each patent of the last time period receives a value for its promisingness expressed as a non-negative count variable (shown in the file "Promisingness_score_t3_sorted.xlsx").

# Note: 
The "STATA_Poisson_regression.txt" can be used to reproduce the statistical assessment of promising patents vs. non-promising patents.

The codes "R1_Train_each_predictor_separately.py" and "R2_Permutation_Importance.py" are not part of the manuscript but were created in the revision process.

The code "R1_Train_each_predictor_separately.py" applies supervised machine learning training and testing to only individual predictors and thus tests the individual predictive power of each predictor.

The code "R2_Permutation_Importance.py" focuses on individual predictors' predictive power, as well. However, it utilizes permutation importance and thus tests to which magnitude the predictive power of the best performing model reduces when the predictors are permuted one-by-one.
