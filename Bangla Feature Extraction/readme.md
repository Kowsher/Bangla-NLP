# Bangla Feature Extractor(BFE)

BFE is a Bangla Natural Language Processing based feature extractor.


## Current Features
---

  1. CountVectorizer
  2. TfIdf
  3. Word Embedding
      * Word2Vec
      * FastText

## Installation
```
pip install bfe
```
## Example
### 3. Word Embedding

### Word2Vec

- **Training**
```py
from "package_name" import BN_Word2Vec
#Training Against Sentences
w2v = BN_Word2Vec(sentences=[['আমার', 'প্রিয়', 'জন্মভূমি'], ['বাংলা', 'আমার', 'মাতৃভাষা']])
w2v.train_Word2Vec()

#Training Against one Dataset
w2v = BN_Word2Vec(corpus_file="path to data or txt file")
w2v.train_Word2Vec()

#Training Against Multiple Dataset
'''
    path
      ->data
        ->1.txt
        ->2.txt
        ->3.txt
'''
w2v = BN_Word2Vec(corpus_path="path/data")
w2v.train_Word2Vec(epochs=25)
```

- **Get Word Vector**
```py
from "package_name" import BN_Word2Vec 
w2v = BN_Word2Vec(model_name='give the model name here')
w2v.get_wordVector('আমার')
```

- **Get Similarity**
```py
from "package_name" import BN_Word2Vec 
w2v = BN_Word2Vec(model_name='give the model name here')
w2v.get_similarity('ঢাকা', 'রাজধানী')

#Output: 67.457879
```

- **Get n Similar Words**
```py
from "package_name" import BN_Word2Vec 
w2v = BN_Word2Vec(model_name='give the model name here')
w2v.get_n_similarWord(['পদ্মা'], n=10)
#Output: 
'''
[('সেতুর', 0.5857524275779724),
 ('মুলফৎগঞ্জ', 0.5773632526397705),
 ('মহানন্দা', 0.5634652376174927),
 ("'পদ্মা", 0.5617109537124634),
 ('গোমতী', 0.5605217218399048),
 ('পদ্মার', 0.5547558069229126),
 ('তুলসীগঙ্গা', 0.5274507999420166),
 ('নদীর', 0.5232067704200745),
 ('সেতু', 0.5225246548652649),
 ('সেতুতে', 0.5192927718162537)]
'''
```

- **Get Middle Word**

    Get the probability distribution of the center word given words list.
```py
from "package_name" import BN_Word2Vec 
w2v = BN_Word2Vec(model_name='give the model name here')
w2v.get_outputWord(['ঢাকায়', 'মৃত্যু'], n=2)

#Output:  [("হয়েছে।',", 0.05880642), ('শ্রমিকের', 0.05639163)]
```

- **Get Odd Words**

    Get the most unmatched word out from given words list
```py
from "package_name" import BN_Word2Vec 
w2v = BN_Word2Vec(model_name='give the model name here')
w2v.get_oddWords(['চাল', 'ডাল', 'চিনি', 'আকাশ'])

#Output: 'আকাশ' 
```

- **Get Similarity Plot**

    Creates a barplot of similar words with their probability 

```py
from "package_name" import BN_Word2Vec 
w2v = BN_Word2Vec(model_name='give the model name here')
w2v.get_oddWords(['চাল', 'ডাল', 'চিনি', 'আকাশ'])
```



