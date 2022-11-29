'''
Skeleton for Homework 4: Logistic Regression and Decision Trees
Part 1: Logistic Regression

Authors: Anja Gumpinger, Dean Bodenham, Bastian Rieck
'''

# !/usr/bin/env python3

import pandas as pd
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
                               solver='lbfgs', )
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
    x_trn = scaler.fit_transform(x_trn)

    # check distribution of y_trn
    # print('Distribution of y_trn: ', np.unique(y_trn, return_counts=True))
    # Train model
    model.fit(x_trn, y_trn)

    # Predict labels
    x_tst = scaler.transform(x_tst)
    y_pred = model.predict(x_tst)

    compute_metrics(y_tst, y_pred)

    print('')
    print('Exercise 1.b')
    print('-' * 12)
    print('For the diabetes dataset, I would choose Logistic Regression over LDA because '
          'the data is not balanced in this case, so LDA will not work well. This is obvious '
          'once you set class_weight to balanced in the LR model: the accuracy drops to 0.777 '
          '(still higher than LDA) and we get fewer false negatives, which is important since '
          'misdiagnosis of diabetes is more dangerous than misdiagnosis of non-diabetes.')

    print('')
    print('Exercise 1.c')
    print('-' * 12)
    print('The performance of the model is indeed due to the class imbalance. If '
          'the dataset were different, it would be hard to tell which model would perform better, '
          'unless we test them; both models have their advantages and disadvantages. If I had to pick,'
          'I would probably try LR first since it is somewhat robust to non-normal distributions '
          'and it can be regularised, all while having a lower computational complexity for the '
          'training objective compared to LDA.')

    print(f"Coefficients: {model.coef_}")
    print("Intercept: ", model.intercept_)
    print('')
    print('Exercise 1.d')
    print('-' * 12)
    print('To analyze coefficients I set the regularizer to L1. This is because L1 '
          'induces sparsity. This allowed me to see that the most important two features '
          'are glucose and the diabetes pedigree function. \nWithout L1 penalty, the coefficient '
          'for npreg is 0.33. Calculating the exponential function '
          'results in 1.38, which amounts to an increase of 39.7 percent per additional pregnancy.')

    # print(math.exp(model.coef_[0][0]))
