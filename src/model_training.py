#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 17:06:13 2017

@author: yaric
"""

import argparse

import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def parse_data_corpus(file):
    labels = []
    docs = []
    with open(file) as f:
        docs_num = int(f.readline())
        for i in range(docs_num):
            l = f.readline()
            l_strs = l.split(" ", maxsplit=1)
            labels.append(int(l_strs[0]))
            docs.append(l_strs[1])
            
    return {"target":labels, "data":docs}
            
            
def build_model_for_corpus(file):
    data_train = parse_data_corpus(file)
    print("Fetched %d documents with %d labels\n" % 
          (len(data_train["data"]), len(data_train["target"])))
    
    X_train = data_train["data"]
    y_train = np.array(data_train["target"], dtype=int)
    
    clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            n_iter=5)),
                    ])
    
    clf.fit(X_train, y_train)
    
    return clf
    
def predict_from_input(clf):
    N = 4
    X_test = ["this is a document", "this is another document", "documents are seperated by newlines", 
              "champion products ch approves stock split champion products inc said its board of directors approved a two for one stock split of its common shares for shareholders of record as of april the company also said its board voted to recommend to shareholders at the annual meeting april an increase in the authorized capital stock from five mln to mln shares reuter"]
    
    lbls = clf.predict(X_test)
    for i in range(N):
        print(lbls[i])
    
def search_optimal_parameters(file):
    data_train = parse_data_corpus(file)
    print("Fetched %d documents with %d labels\n" % 
          (len(data_train["data"]), len(data_train["target"])))
    
    X = data_train["data"]
    y = np.array(data_train["target"], dtype=int)
    
    clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, random_state=42, 
                                            n_iter=5)),
                    ])
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'vect__analyzer': ['word', 'char', 'char_wb'],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-3, 1e-4),
                 }
    gs_clf = GridSearchCV(clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(X, y)
    
    
    print("Best score: %.2f" % (gs_clf.best_score_))
    
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
        
    return gs_clf
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpora', help = 'the data corpus file')
    args = parser.parse_args()
    
    print("Building model for %s" % (args.corpora))
    
    clf = search_optimal_parameters(args.corpora)
    
    predict_from_input(clf)
