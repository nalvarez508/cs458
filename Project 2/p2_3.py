import numpy as np
from sklearn import datasets

# (a) load newsgroups
ng_train = datasets.fetch_20newsgroups(subset='train', categories=['rec.autos', 'talk.religion.misc', 'comp.graphics', 'sci.space'])
ng_test = datasets.fetch_20newsgroups(subset='test', categories=['rec.autos', 'talk.religion.misc', 'comp.graphics', 'sci.space'])

print(f'''
Set\t_|_ # Docs\t_|_ Attributes
Train\t |  {ng_train.target.shape}\t |  ?
Test\t |  {ng_test.target.shape}\t |  N/A''')

