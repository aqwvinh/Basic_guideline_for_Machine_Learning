# Build custom transformers pipeline for preprocessing

Tips:
- Syntax is class DerivedClass(BaseClass):
                   BodyDerivedClass
- We use inheritance from two base classes **TransformerMixin** and **BaseEstimator** ==> to get fit/transform and get/set params for free
- fit_transform on train and transform on test


## Pipeline example
```
import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
```
