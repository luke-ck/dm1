#!/usr/bin/env python3

'''
Skeleton for Homework 4: Logistic Regression and Decision Trees
Part 2: Decision Trees

Authors: Anja Gumpinger, Bastian Rieck
'''
from math import floor, ceil

import numpy as np
from sklearn.datasets import load_iris


def compute_information_content(y):
    '''
     compute the information content of a subset X with labels y
    defined as - P(y=1|x) log_2 P(y=1|x) - P(y=0|x) log_2 P(y=0|x)
    :param y: class
    :param y: class labels
    :return: information content
    '''
    classes = np.unique(y)
    info_content = 0
    for c in classes:
        p = len(y[y == c]) / len(y)
        info_content -= p * np.log2(p)

    return info_content


def split_data(X, y, attribute_index, theta):
    """
    function that splits X and y into two subsets according to the split
    defined by the pair (attribute_index, theta)
    :param X: feature matrix
    :param y: class labels
    :param attribute_index: index of the attribute to split on, ranges between 0 and 3
    :param theta: threshold to split on
    :return:
        two subsets of X and y
    """
    # split based on the attribute_index and theta
    # return the two subsets of X and y

    y_left = y[X[:, attribute_index] < theta]
    y_right = y[X[:, attribute_index] >= theta]

    return y_left, y_right


def compute_information_gain(X, y, attribute_index, theta):
    '''
    Computes the information gain for a dataset X with labels y that is split according to the
    split defined by the pair (attribute_index, theta).

    :param X: feature matrix
    :param y: class labels
    :param attribute_index: index of the attribute to split on
    :param theta: threshold to split on
    :return: information gain
    '''

    # compute the information gain for the split defined by (attribute_index, theta)
    # return the information gain
    y_left, y_right = split_data(X, y, attribute_index, theta)
    info_gain = compute_information_content(y)

    for y_subset in [y_left, y_right]:
        info_gain -= len(y_subset) / len(y) * compute_information_content(y_subset)

    return info_gain


def find_best_split(X, y):
    '''
    Finds the best split for a dataset X with labels y.

    :param X: feature matrix
    :param y: class labels
    :return: best split (attribute_index, theta)
    '''

    # find the best split for the dataset X with labels y
    # return the best split (attribute_index, theta)
    best_gain = 0
    best_split = (0, 0)
    for i in range(X.shape[1]):
        for theta in np.unique(X[:, i]):
            info_gain = compute_information_gain(X, y, i, theta)
            if info_gain > best_gain:
                best_gain = info_gain
                best_split = (i, theta)

    return best_split


def cross_validation(X, y, k, model):
    """
    function that performs k-fold cross validation on the dataset X with labels y
    :param X: Feature matrix
    :param y: class labels
    :param k: number of folds
    :return:
        mean accuracy of the k-fold cross validation
    """

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    return np.mean(accuracies)


def report_important_features(model, feature_names):
    """
    function that prints the most important features of a decision tree
    :param model: decision tree model
    :return:
    """
    print("Important features:")
    feature_importances = np.argsort(model.feature_importances_)[::-1]
    for i, feature in enumerate(feature_importances):
        print("{}. {} ({})".format(i + 1, feature_names[feature], model.feature_importances_[feature]))
    print('')


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target

    feature_names = iris.feature_names
    num_features = len(set(feature_names))

    ####################################################################
    # Your code goes here.
    ####################################################################

    # compute the information content of the whole dataset
    info_content = compute_information_content(y)
    print('Information content of the whole dataset: {0:.3f}'.format(info_content))

    attributes_to_try = [0, 1, 2, 3]
    theta_to_try = [5.5, 3.0, 2.0, 1.0]
    print('Exercise 2.b')
    print('------------')
    for i, theta in zip(attributes_to_try, theta_to_try):
        information_gain = compute_information_gain(X, y, i, theta)
        print('split on attribute {0}, threshold {1:.1f}: Information gain = {2:.2f}'.format(
            feature_names[i], theta, information_gain))

    print('')

    print('Exercise 2.c')
    print('I would choose to split on attribute {0}, threshold {1:.1f} since it has the ' \
          'highest information gain.'.format(feature_names[2], 2.0))

    print('')

    ####################################################################
    # Exercise 2.d
    ####################################################################

    # Do _not_ remove this line because you will get different splits
    # which make your results different from the expected ones...
    np.random.seed(42)

    from sklearn.model_selection import KFold
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    # split the data into k folds
    k = 5
    clf = DecisionTreeClassifier()

    # compute the accuracy of the classifier on the test set
    # for each fold
    accuracy = cross_validation(X, y, k, clf)

    print('Accuracy score using cross-validation')
    print('-------------------------------------\n')
    print('Mean accuracy: {0:.2f}'.format(accuracy * 100))

    print('')
    print('Feature importances for _original_ data set')
    print('-------------------------------------------\n')
    report_important_features(clf, feature_names)

    # remove the label 2
    X = X[y != 2]
    y = y[y != 2]

    clf = DecisionTreeClassifier()
    accuracy = cross_validation(X, y, k, clf)
    print('Feature importances for _reduced_ data set')
    print('Mean accuracy: {0:.2f}'.format(accuracy * 100))
    print('------------------------------------------\n')
    report_important_features(clf, feature_names)
