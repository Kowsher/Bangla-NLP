# Bangla Feature Extractor(BFE)
===================================
BFE is a Bangla Natural Language Processing based feature extractor.


# Current Features
---

  - CountVectorizer
  - TfIdf
  - Word Embedding
    - Word2Vec
    - FastText

## Installation
```
pip install bfe
```
## Example
### Word2Vec
- Training
```py
from "package_name" import BN_Word2Vec
#Training Against Sentences
w2v = BN_Word2Vec(sentences=[['আমার', 'প্রিয়', 'জন্মভূমি'], ['বাংলা', 'আমার', 'মাতৃভাষা']])
w2v.train_Word2Vec()

#Training Against one Dataset
w2v = BN_Word2Vec(corpus_file="path to data or txt file")
w2v.train_Word2Vec()

#Training Against Multiple Dataset
    path
      ->data
        ->1.txt
        ->2.txt
        ->3.txt
>>> w2v = BN_Word2Vec(corpus_path="path/data")
>>> w2v.train_Word2Vec(epochs=25)
```