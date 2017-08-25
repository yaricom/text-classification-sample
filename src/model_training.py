#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 17:06:13 2017

@author: yaric
"""

import argparse

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier

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
    
    #vectorizer = TfidfVectorizer(sublinear_tf=False, max_df=0.5, stop_words='english')
    vectorizer = CountVectorizer(stop_words='english')
    #vectorizer = HashingVectorizer(stop_words='english',  n_features=2 ** 16)
    X_train = vectorizer.fit_transform(data_train["data"])
    print("n_samples: %d, n_features: %d" % X_train.shape)
    
    y_train = np.array(data_train["target"], dtype=int)
    print(y_train.shape)
    
    clf = SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet")
    clf = KNeighborsClassifier(n_neighbors=10)
    
    clf.fit(X_train, y_train)
    
    return vectorizer, clf
    
def predict_from_input(vectorizer, clf):
    N = 4
    test = ["this is a document", "this is another document", "documents are seperated by newlines", "champion products ch approves stock split champion products inc said its board of directors approved a two for one stock split of its common shares for shareholders of record as of april the company also said its board voted to recommend to shareholders at the annual meeting april an increase in the authorized capital stock from five mln to mln shares reuter"]
    #test = np.array([input_str for _ in range(N)], str)
    X_test = vectorizer.transform(test)
    print("n_samples: %d, n_features: %d" % X_test.shape)
    lbls = clf.predict(X_test)
    for i in range(N):
        print(lbls[i])
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('corpora', help = 'the data corpus file')
    args = parser.parse_args()
    
    print("Building model for %s" % (args.corpora))
    
    vectorizer, clf = build_model_for_corpus(args.corpora)
    
    predict_from_input(vectorizer, clf)
