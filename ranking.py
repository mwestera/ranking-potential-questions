#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AM: Questions and Models of Discourse
author: Luise Schricker

python-3.4

sklearn-0.19.1

This file contains a class with methods for ranking a list of potential questions, given the preceding assertion
for my project "Ranking potential Questions".
"""
import json
import os.path
import random
import sys

from sklearn.externals import joblib
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from extract_features import FeatureExtractor

class QuestionRanker():
    """
    Class that has methods for ranking a list of potential questions, given the preceding assertion.
    """

    def __init__(self, word_vecs, train=False, train_path=None):
        """
        :param word_vecs: path to word vectors
        :param train: indicates whether the RandomForest Classifier should be retrained
        """

        self.Fe = FeatureExtractor(word_vecs)
        random.seed(5)

        # Retrain the RandomForest Classifier if train is set to True
        if train:
            if train_path==None:
                print("ERROR: Missing path to training data. Please set parameter train_path to the path to the training data.")
                exit(1)
            else:
                self.clf = self.train_random_forest(train_path)

        else:
            # Load the existing classifier
            self.clf = joblib.load('model/model.pkl')


    def rank_questions(self, assertion, potential_questions, mode='uniform'):
        """
        Method for ranking potential questions, given the preceding assertion.

        :param assertion: the preceding assertion
        :param potential_questions: a list of potential questions
        :param mode: possible values are 'uniform' and 'ml' for ranking with uniform feature weights
                    and machine learning weights.
        :return: a list containing the questions in the order of the ranking. The list contains
                tuples of the question and the ranking value.
        """
        question_tuples = []

        # Generate question-assertion pairs
        for question in potential_questions:
            if mode=='uniform':
                ranking_score = self.Fe.transform_scalar(assertion, question)
                question_tuples.append((question, ranking_score))
            elif mode=='ml':
                # predict_proba returns the class probabilities
                class_scores = self.clf.predict_proba([self.Fe.transform_vector(assertion, question)])[0]

                # Find the correct score for class 1
                for i, cl in enumerate(self.clf.classes_):
                    if cl == 1:
                        question_tuples.append((question, class_scores[i]))

        # Sort questions by ranking score
        ranked_questions = sorted(question_tuples, key=lambda x: x[1], reverse=True)

        return ranked_questions

    def evaluate(self, path_data, mode='uniform', top=1):
        """
        Method to evaluate the question ranker.

        :param path_data: path to the data-file, which should hold the test data in json format
                        (see extract_data.py).
        :param mode: Mode of the ranker to evaluate, possible values are 'baseline', 'uniform' and 'ml'.
        :param top: Number of top ranks amongst which one of the labels has to appear in order
                    to be considered correct.
        :return:
        """
        print("Evaluating the ranker with mode '{}'.\n".format(mode))

        with open(path_data, "r", encoding='utf-8') as f:
            data = json.load(f)

        y_true = [1 for datapoint in data]
        y_pred = []

        for assertion in data:
            print("Assertion: ", assertion)
            potential_questions = data[assertion]['potential_qs']
            labels = data[assertion]['labels']

            if mode in ['uniform', 'ml']:
                ranked_questions = self.rank_questions(assertion, potential_questions, mode=mode)
            # For the baseline, order the potential questions randomly
            elif mode=='baseline':
                ranked_questions = potential_questions
                random.shuffle(ranked_questions)
                ranked_questions = [(pq, 0) for pq in ranked_questions]

            # 1 := one of the labels is among the top ranked questions,
            # 0 := one of the labels is not among the top ranked questions
            top_ranks = [tup[0] for tup in ranked_questions[:top]]

            pred_true = False
            for label in labels:
                if label in top_ranks:
                    y_pred.append(1)
                    pred_true = True
                    break
            if not pred_true:
                y_pred.append(0)

            print("Top ranked question: {} {}".format(top_ranks[0], "CORRECT" if pred_true else ""))

        # Use sklearn.metrics.accuracy_score to compute the accuracy
        print("\nAccuracy: {}\n".format(accuracy_score(y_true, y_pred)))
        return

    def inspect_features(self):
        """
        Method for inspecting the features of the Random Forest Classifier by importance.
        """
        features = [foo.__name__ for foo in self.Fe.generation_priciples + self.Fe.ordering_principles + self.Fe.qud_constraints]
        importances = self.clf.feature_importances_

        print("Features by importance:\n*******************\n")

        for tup in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
            print("Feature: {}, Importance: {}".format(tup[0], tup[1]))

    def train_random_forest(self, train_path):
        """
        Train a Random Forest Classifier to be used in ml based ranking.

        :param train_path: path to the training data
        :return: the trained classifier
        """
        print("Training a Random Forest Classifier for ml based ranking.")

        # Check if the training data was already transformed to feature represenations
        path_train_features = "".join(train_path.split(".")[:-1] + ["_features.", train_path.split(".")[-1]])
        if os.path.isfile(path_train_features):
            with open(path_train_features, "r", encoding='utf-8') as f:
                data = json.load(f)
                X = []
                y = []

                for datapoint in data:
                    X.append(datapoint["X"])
                    y.append(datapoint["y"])

        else:
            # Load training data
            with open(train_path, "r", encoding='utf-8') as f:
                data = json.load(f)

            X = []
            y = []

            counter = 0
            data_out = []
            print("Preprocessing training data ...")
            for datapoint in data:
                d = {}

                # Append vector representation and label of correct question-assertion pair
                X_cur = self.Fe.transform_vector(datapoint["assertion"], datapoint["question"])
                X.append(X_cur)
                y.append(1)
                d["X"] = X_cur
                d["y"] = 1
                data_out.append(d)

                # Append vector representation and label of incorrect question-assertion pair
                for qu in datapoint["alternative_questions"]:
                    d = {}
                    X_cur = self.Fe.transform_vector(datapoint["assertion"], qu)
                    X.append(X_cur)
                    y.append(0)
                    d["X"] = X_cur
                    d["y"] = 0
                    data_out.append(d)

                counter += 1

                # Write a progress bar to stdout to track progress
                sys.stdout.write('\r')
                sys.stdout.write(
                    "[%-20s] %d%%" % ('=' * (counter // (len(data) // 100)), ((counter + 1) / (len(data) / 100))))
                sys.stdout.flush()

            # Save preprocessed data to file
            with open(path_train_features, "w", encoding='utf-8') as f_out:
                json.dump(data_out, f_out, indent=4)

        clf = RandomForestClassifier(random_state=5, min_samples_leaf=5, max_depth=10, class_weight={0:0.5, 1:1})

        print("\nTraining...")
        clf.fit(X, y)

        print("Accuracy on training set: {}".format(clf.score(X,y)))

        # Save the trained classifier
        joblib.dump(clf, 'model/model.pkl')
        print("Done.\n")

        return(clf)

# Execution
if __name__ == "__main__":
    # Example usage
    ranker = QuestionRanker(word_vecs="word2vec/GoogleNews-vectors-negative300.bin",
                            train=True,
                            train_path="swda_train.json")
    ranker.evaluate("snowden_qud_dev.json", mode='ml', top=1)
    ranker.evaluate("snowden_qud_dev.json", mode='uniform', top=3)
    ranker.evaluate("snowden_qud_dev.json", mode='baseline', top=5)
    ranker.inspect_features()
    print(ranker.rank_questions("Hello, my name is Luise.", ["Who are you?",
                                                       "What do you like to do?",
                                                       "How is the weather?",
                                                       "Where are you from?",
                                                       "Do you have a dog?"]))

