Bangla Lemmatizer 
====================
Natural language processing (NLP) finds enormous applications in automation, while lemmatization is an essential preprocessing technique for simplification of a word to its origin-word in NLP. However, there is a scarcity of effective algorithms in Bangla NLP. This leads us to develop a useful Bangla language lemmatization tool. Usually, some rule base stemming processes play the vital role of lemmatization in Bangla language processing for the lack of any lemmatization tool. In this paper, we propose a Bangla lemmatization framework as well as three effective lemmatization techniques based on data structures and algorithms that are used to develop the framework. In this paper, we have used Trie algorithm based on prefixes in Bangla language and developed a mapping algorithm named “Dictionary Based Search by Removing Affix (DBSRA)” whereas the Levenshtein distance gets the priority of comparison together with unknown words lemmatize than lemmatize of known words. Finally, we have done experimentation for Bangla language lemmatization, and our developed DBSRA confirms better performance in comparison to other algorithms, as well as shown the results of the framework which confirm the best performance with optimal time and space complexity.

This is a reserach and aim to make strong  the Bangla NLP which is going to publish in the 9th ICIEV,IEEE,Japan, 2020. 

Bangla Lemmatizer (BnLemma)
==================================
BnLemma is a powerful tool for getting the root words of the words used in any Bangla sentence.
There is a pretrained model available with this package.
This page gives a basic introduction to the package.

## Installation:

```
pip install BnLemma
```

In Python a Bangla Linga is typically written as:

## Usage:
```python
>>>  from BnLemma import lemmatization as lm
>>>  s = "মানুষের জীবনটা পাচ্ছেন তাই কাজে লাগানো দরকার আমাদেরকে"  
>>>  s = lm.lemma(s)
>>>  print(s)
       
```
Output:
```
মানুষ জীবন পাওয়া তাই কাজ লাগা দরকার আমাদের
```
