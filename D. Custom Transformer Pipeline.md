# Build custom transformers pipeline for preprocessing

Tips:
- Syntax is class DerivedClass(BaseClass):
                   BodyDerivedClass
- We use inheritance from two base classes **TransformerMixin** and **BaseEstimator** ==> to get fit/transform and get/set params for free
- fit_transform on train and transform on test. Function fit only returns self so no interest here
- Useful to build 2 pipelines: one for categorical and one for numerical
- Combine two pipelines using **FeatureUnion** class from scikit-learn. Bonus: parrallel computation direct with FeatureUnion so faster than linear execution
- Important: FeatureUnion only takes transformers, we can't add any ML model


## Pipeline example
```
## Import librairies
import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline

## Custom Transformer that extracts columns passed as argument to its constructor 
class FeatureSelector(TransformerMixin, BaseEstimator):
    #Class Constructor 
    def __init__( self, feature_names ):
        self._feature_names = feature_names 
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do: select feature_names
    def transform( self, X, y = None ):
        return X[ self._feature_names ] 

## CATEGORICAL PIPELINE
#Custom transformer that breaks dates column into year, month and day into separate columns and converts certain features to binary
# Original date format is 'YYYYMMDDT000000'

class CategoricalTransformer( BaseEstimator, TransformerMixin ):
    #Class constructor method that takes in a list of values as its argument
    def __init__(self, use_dates = ['year', 'month', 'day'] ):
        self._use_dates = use_dates
        
    #Return self nothing else to do here
    def fit( self, X, y = None  ):
        return self

    #Helper function to extract year from column 'dates' 
    def get_year( self, obj ):
        return str(obj)[:4]
    
    #Helper function to extract month from column 'dates'
    def get_month( self, obj ):
        return str(obj)[4:6]
    
    #Helper function to extract day from column 'dates'
    def get_day(self, obj):
        return str(obj)[6:8]
    
    #Helper function that converts values to Binary depending on input 
    def create_binary(self, obj):
        if obj == 0:
            return 'No'
        else:
            return 'Yes'
    
    #Transformer method we wrote for this transformer 
    def transform(self, X , y = None ):
       #Depending on constructor argument break dates column into specified units
       #using the helper functions written above 
       for spec in self._use_dates:
        
        exec( "X.loc[:,'{}'] = X['date'].apply(self.get_{})".format( spec, spec ) )
       #Drop unusable column 
       X = X.drop('date', axis = 1 )
       
       #Convert these columns to binary for one-hot-encoding later
       X.loc[:,'waterfront'] = X['waterfront'].apply( self.create_binary )
       
       X.loc[:,'view'] = X['view'].apply( self.create_binary )
       
       X.loc[:,'yr_renovated'] = X['yr_renovated'].apply( self.create_binary )
       #returns numpy array
       return X.values 

## NUMERICAL PIPELINE
# Compute bathrooms per bedroom and/or (function associated with a boolean value) how old the house is in 2020

class NumericalTransformer(BaseEstimator, TransformerMixin):
    #Class Constructor
    def __init__( self, bath_per_bed = True, years_old = True ):
        self._bath_per_bed = bath_per_bed
        self._years_old = years_old
        
    #Return self, nothing else to do here
    def fit( self, X, y = None ):
        return self 
    
    #Custom transform method we wrote that creates aformentioned features and drops redundant ones 
    def transform(self, X, y = None):
        #Check if needed 
        if self._bath_per_bed:    # if bath_per_bed = True
            #create new column
            X.loc[:,'bath_per_bed'] = X['bathrooms'] / X['bedrooms']
            #drop redundant column
            X.drop('bathrooms', axis = 1 )
        #Check if needed     
        if self._years_old:
            #create new column
            X.loc[:,'years_old'] =  2020 - X['yr_built']
            #drop redundant column 
            X.drop('yr_built', axis = 1)
            
        #Converting any infinity values in the dataset to Nan
        X = X.replace( [ np.inf, -np.inf ], np.nan )
        #returns a numpy array
        return X.values


## COMBINE PIPELINES

#Categrical features to pass down the categorical pipeline 
cat_features = ['date', 'waterfront', 'view', 'yr_renovated']

#Numerical features to pass down the numerical pipeline 
num_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                'condition', 'grade', 'sqft_basement', 'yr_built']

#Defining the steps in the categorical pipeline 
categorical_pipeline = Pipeline( steps = [ ( 'cat_selector', FeatureSelector(cat_features) ),  #FeatureSelector is the first transformer we created
                                  
                                  ( 'cat_transformer', CategoricalTransformer() ), 
                                  
                                  ( 'one_hot_encoder', OneHotEncoder( sparse = False ) ) ] ) #We can do MV imputation too
    
#Defining the steps in the numerical pipeline     
numerical_pipeline = Pipeline( steps = [ ( 'num_selector', FeatureSelector(num_features) ),
                                  
                                  ( 'num_transformer', NumericalTransformer() ),
                                  
                                  ('imputer', SimpleImputer(strategy = 'median') ),
                                  
                                  ( 'std_scaler', StandardScaler() ) ] )

#Combining numerical and categorical piepline into one full big pipeline horizontally using FeatureUnion
full_pipeline = FeatureUnion( transformer_list = [ ( 'categorical_pipeline', categorical_pipeline ), 
                                                  
                                                  ( 'numerical_pipeline', numerical_pipeline ) ] )


## FINAL PIPELINE WITH ML MODEL
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Input of the pipeline is a dataframe and we exclude price
X = data.drop('price', axis = 1)
#You can covert the target variable to numpy 
y = data['price'].values 

# Apply preprocessing on train
X_train, X_test, y_train, y_test = train_test_split( X, y , test_size = 0.2 , random_state = 94 )

#The full pipeline as a step in another pipeline with an estimator as the final step
final_pipeline = Pipeline( steps = [ ( 'full_pipeline', full_pipeline),
                                  
                                  ( 'model', LinearRegression() ) ] )

#Can call fit on it just like any other pipeline
final_pipeline.fit( X_train, y_train )

#Can predict with it like any other pipeline
y_pred = final_pipeline.predict( X_test ) 

```
