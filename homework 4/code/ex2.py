#!/usr/bin/env python3

'''
Skeleton for Homework 4: Logistic Regression and Decision Trees
Part 2: Decision Trees

Authors: Anja Gumpinger, Bastian Rieck
'''
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import clone

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
        p = len(y[y == c]) / len(y)  # this is just the frequency
        info_content -= p * np.log2(p)

    return info_content


def split_data(X, y, attribute_index, theta):
    """
    function that splits X and y into two subsets according to the split
    defined by the pair (attribute_index, theta)
    :param X: feature matrix
    :param y: class labels
    :param attribute_index: index of the attribute to split on, ranges between 0 and 3
    :param theta: thetaold to split on
    :return:
        two subsets of X and y
    """

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
    :param theta: thetaold to split on
    :return: information gain
    '''

    # compute the information gain for the split defined by (attribute_index, theta)
    # return the information gain
    y_left, y_right = split_data(X, y, attribute_index, theta)
    info_gain = compute_information_content(y)

    for y_subset in [y_left, y_right]:
        info_gain -= len(y_subset) / len(y) * compute_information_content(y_subset)

    return info_gain


def cross_validation(X, y, k, model):
    """
    function that performs k-fold cross validation for a model on the dataset X with labels y
    :param X: Feature matrix
    :param y: class labels
    :param k: number of folds
    :param model: model to evaluate
    :return:
        mean accuracy
        feature scores with indices of the features sorted by importance
    """

    kf = KFold(n_splits=k, shuffle=True)
    accuracies = []
    feature_scores = []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = clone(model)  # make sure to train on a fresh model

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        feature_scores.append(model.feature_importances_)

    # now we compute the mean over the folds and sort
    accuracies = np.mean(accuracies)
    mean_feature_scores = np.mean(feature_scores, axis=0)
    feature_importances = np.argsort(mean_feature_scores)[::-1]
    feature_importances = list(zip(feature_importances, mean_feature_scores[feature_importances]))

    return accuracies, feature_importances


def report_important_features(feature_names, feature_scores, top_k=2):
    """
    function that prints the most important features of a decision tree
    :param model: decision tree model
    :return:
    """
    for idx, feature in feature_scores[:top_k]:
        if feature == 0:
            continue
        print("{}-({:.3f})".format(feature_names[idx], feature))


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
    print('-' * 12)
    for i, theta in zip(attributes_to_try, theta_to_try):
        information_gain = compute_information_gain(X, y, i, theta)
        print('split on attribute {0}, threshold {1:.1f}: Information gain = {2:.2f}'.format(
            feature_names[i], theta, information_gain))

    print('')

    print('Exercise 2.c')
    print('-' * 12)
    print('I would choose to split on attribute {0} < {2:.1f} or attribute {1} '
          '< {3:.1f} since they have the highest information gain of 0.92.'
          .format(feature_names[2], feature_names[3], theta_to_try[2], theta_to_try[3]))

    print('')

    ####################################################################
    # Exercise 2.d
    ####################################################################
    print('Exercise 2.d')
    print('-' * 12)
    # Do _not_ remove this line because you will get different splits
    # which make your results different from the expected ones...
    np.random.seed(42)

    # split the data into k folds
    k = 5
    clf = DecisionTreeClassifier()

    # compute the accuracy of the classifier on the test set
    # for each fold
    accuracy, feature_scores = cross_validation(X, y, k, clf)

    print('Accuracy score using cross-validation')
    print('-' * 12)
    print('Mean accuracy: {0:.2f}'.format(accuracy * 100))

    print('')
    print('Feature importances for _original_ data set')
    print('-' * 12)
    report_important_features(feature_names, feature_scores)
    # remove the label 2
    X = X[y != 2]
    y = y[y != 2]

    clf = DecisionTreeClassifier()
    accuracy, feature_scores = cross_validation(X, y, k, clf)

    print('')
    print('Feature importances for _reduced_ data set')
    # print('Mean accuracy: {0:.2f}'.format(accuracy * 100))
    print('-' * 12)
    report_important_features(feature_names, feature_scores)
    print('')
    print('Feature importance is calculated as a measure of the weighted decrease in node impurity.'
          ' The feature score of 1 means that splitting on one feature is enough to classify the (binary) data.')
