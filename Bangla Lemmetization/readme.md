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
>>>  from BnLemma import lemmatization
>>>  s = "মানুষের জীবনটা পাচ্ছেন তাই কাজে লাগানো দরকার আমাদেরকে"  
>>>  s = lm.lemma(s)
>>>  print(s)
       
```
Output:
```
মানুষ জীবন পাওয়া তাই কাজ লাগা দরকার আমাদের
```
