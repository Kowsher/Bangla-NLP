# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:22:39 2020

@author: Kowsher
"""

from scipy.sparse import csr_matrix
import pickle
from pathlib import Path
script_location = Path(__file__).absolute().parent
wordset_loc = script_location / "word_set"

class CountVectorizer(object):
    
    def __init__(self, word_set = wordset_loc):
        self.sz = 0
        self.wordSet = set()
        self.word_set = word_set
         
    def fit_transform(self,doc):
        ln = len(doc)
        for i in doc:
            i = i.split(" ")
            self.wordSet = set(self.wordSet).union(set(i))
        
        l = list(self.wordSet)
        with open(self.word_set, 'wb') as p:
          pickle.dump(l, p)
        
        self.sz = len( self.wordSet)
        matrix = csr_matrix((ln, self.sz)).toarray()
        dc_ln = 0
        for i in doc:
            i = i.split(" ")
            w_ln = 0
            
            for j in  self.wordSet:
                
                cnt = 0
                for k in i:
                    if j == k:
                        cnt = cnt + 1
                matrix[dc_ln][w_ln] = cnt
                w_ln = w_ln + 1
            dc_ln = dc_ln + 1  
        
        return matrix
    
    def get_wordSet(self):
        return self.wordSet
    
    def transform(self, pat):
      
        with open(wordset_loc, 'rb') as p:
          wordSet_m = pickle.load(p)
        #print(len(self.wordSet))
        
        sz_m = len(wordSet_m)
        
        ln_pat = len(pat)
        matrix_pat = csr_matrix((ln_pat, sz_m)).toarray()
        dc_ln = 0
        for i in pat:
            i = i.split(" ")
            w_ln = 0
            
            for j in wordSet_m:
                cnt = 0
                for k in i:
                    if j == k:
                        cnt = cnt + 1
                matrix_pat[dc_ln][w_ln] = cnt
                w_ln = w_ln + 1
            dc_ln = dc_ln + 1
            
        return matrix_pat