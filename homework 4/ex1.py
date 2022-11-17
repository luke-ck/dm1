'''
Skeleton for Homework 4: Logistic Regression and Decision Trees
Part 1: Logistic Regression

Authors: Anja Gumpinger, Dean Bodenham, Bastian Rieck
'''
import math

#!/usr/bin/env python3

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


def compute_metrics(y_true, y_pred):
    '''
    Computes several quality metrics of the predicted labels and prints
    them to `stdout`.

    :param y_true: true class labels
    :param y_pred: predicted class labels
    '''

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print('Exercise 1.a')
    print('------------')
    print('TP: {0:d}'.format(tp))
    print('FP: {0:d}'.format(fp))
    print('TN: {0:d}'.format(tn))
    print('FN: {0:d}'.format(fn))
    print('Accuracy: {0:.3f}'.format(accuracy_score(y_true, y_pred)))


if __name__ == "__main__":

    ###################################################################
    # Your code goes here.
    ###################################################################
    model = LogisticRegression(random_state=69,
                               max_iter=1000,
                               # penalty='l1',
                               solver='saga',)
                               # class_weight='balanced')
    scaler = StandardScaler()

    datadir = 'data/'
    # Load data
    df_train = pd.read_csv(datadir + 'diabetes_train.csv')
    df_test = pd.read_csv(datadir + 'diabetes_test.csv')

    # Split into features and labels
    x_trn = df_train.drop('type', axis=1).values
    y_trn = df_train['type'].values
    x_tst = df_test.drop('type', axis=1).values
    y_tst = df_test['type'].values

    # Scale features
    x_trn = scaler.fit_transform(x_trn, y_trn)

    # check distribution of y_trn
    # print('Distribution of y_trn: ', np.unique(y_trn, return_counts=True))
    # Train model
    model.fit(x_trn, y_trn)

    # Predict labels
    x_tst = scaler.transform(x_tst)
    y_pred = model.predict(x_tst)

    compute_metrics(y_tst, y_pred)

    print(model.coef_)
    print(model.intercept_)

    # compute the log odds for attribute 1

    # print('Exercise 1.b')
    # print('I would choose Logistic Regression for this particular dataset over LDA because
    # logistic Regression is more robust to non-normal distributions. Also, the data is
    # not balanced in this case, so LDA will not work well. In the case where the
    # parameter class_weight is set to balanced, logistic regression still performs better than
    # LDA.')
    #
    # print('Exercise 1.c')
    # print('The performance of the model is indeed because of the class imbalance. If
    # the dataset were different, it would be hard to tell which model would perform better; both
    # models have their advantages and disadvantages.')

    # print('Exercise 1.d')
    # print('To analyze coefficients I set the regularizer to L1. This is because L1
    # induces sparsity. This allowed me to see that the most important two features
    # glucose and the diabetes pedigree function. Without L1 penalty the coefficient for
    # the number of pregnancies npreg was 3.34e-1, and the odds of having diabetes
    # increased by 1.387 times for every additional pregnancy.') TODO: add percentage instead

    print(math.exp(model.coef_[0][0]))

