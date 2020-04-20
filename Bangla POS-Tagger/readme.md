Ekushey Bangla POS(Parts of Speech) Tagger
==================================
Ekushey Bangla POS Tagger is a powerful tool for annotating parts of speech of any Bangla sentence.
There is a pretrained model available with this package, trained with viterbi algorithm.
This page gives a basic introduction to the package.

## Installation:

```
pip install ekushey
```

In Python a Bangla Linga is typically written as:

## Usage:
```python
>>>  from ekushey.pos_tagger import pos_tagger
>>>  pos = pos_tagger()
>>>  pos.tag("আমার একটি কলম আছে")
        
```
```
Output:
('আমার', 'PRP'), ('একটি', 'QC'), ('কলম', 'Unk'), ('আছে', 'VM')]
```
