Bangla Linga - Python based Gender prediction from Bangla name (bangla-linga)
==========================

Bangla Linga is a powerful tool for predicting gender/linga from Bengali name patterns.
This page gives a basic introduction to the package.

## Installation:

```
pip install bangla-linga
```

In Python a Bangla Linga is typically written as:

## Usage:
```python
>>> from bangla_linga import gender_prediction as gp
>>> gen = gp.BN_gen_pred()
>>> gen.predict_gender("রাহাত")
```
## Output:
'Male'
