# Feature Engineering

## Typical steps and tips for features engineering (not comprehensive)
- Check variables type : numerical (discrete/continuous), categorical (ordinal/nominal), date and mixed variables (numbers and labels in the same observation like car numberplate or in ≠ observations)
- Check MV (understand the mechanisms - MCAR/MNAR/MAR - to choose the right imputation technique - mean/median/mode/arbitrary/frequent/indicator/random/knn etc)
- Check cardinality (high cardinaly may cause overfitting and operationalisation problems)
- Check rare labels (Same pb than with high cardinality. Regroup rare labels may improve the performance)
- Encoding (OneHot/Label/CountFrequency/Mean/WoE etc)
- Binning/Discretization (create intervals for continuous vars - EqualWidth/EqualFrequency/DecisionTree)
- Outliers (Detection with IQR, remove or treat as MV or discretization or capping=fix limit values. Pb: distorts var distribution)
- Scaling (Standardization/MinMaxScaling/RobustScaler etc)

Once all these steps are done and we've identified our FeatEngin steps, 
- Pipeline

TIPS
- MV: When ≤5%, replace by mean/median and mode for cat vars. When >5%, mean/median and ADD binary missing indicator for num and "Missing" label for cat
- Discretization: Find the optimal intervals using discretization with decision trees cuz create a monotonic relationship (re-order discreted vars) ==> good with 
    linear models and no need to do discretization then ordinal cat
- Scaling: RobustScaler ignores outliers for standardization. Important step EXCEPT FOR TREE --> NO NEED TO DO SCALING FOR TREE
- Encoding: Better to fit on the train only but acceptable to fit on test if we know all the values. NO SCALING ON TEST





## Pipeline example using feature-engine library
```
# for feature engineering
from sklearn.preprocessing import StandardScaler
from feature_engine import missing_data_imputers as mdi
from feature_engine import discretisers as dsc
from feature_engine import categorical_encoders as ce


# First, make a list of the num and cat variables
categorical = [var for var in df.columns if df[var].dtype=='O']
numerical = [var for var in df.columns if df[var].dtype!='O']
# list of variables that contain year information for num vars that are not treated as num
year_vars = [var for var in numerical if 'Yr' in var or 'Year' in var]
# find discrete variables : To identify discrete variables, select from all the numerical ones, those that contain a finite and small number of distinct values.
discrete = []
for var in numerical:
    if len(data[var].unique()) < 20 and var not in year_vars:
        print(var, ' values: ', data[var].unique())
        discrete.append(var)

house_pipe = Pipeline([

    # missing data imputation
    ('missing_ind', mdi.AddNaNBinaryImputer(  # add missing indicator
        variables=['LotFrontage', 'MasVnrArea',  'GarageYrBlt'])),
    ('imputer_num', mdi.MeanMedianImputer(imputation_method='median',  # replace MV by the median
                                          variables=['LotFrontage', 'MasVnrArea',  'GarageYrBlt'])),
    ('0_imputer', mdi.ArbitraryNumberImputer(arbitrary_number = 0,
                                             variables = ['platform_fees', 'cleaning_fees'])),                                          
    ('imputer_cat', mdi.CategoricalVariableImputer(variables=categorical)),  # add string 'missing' to all cat var with MV

    # categorical encoding
    ('rare_label_enc', ce.RareLabelCategoricalEncoder(
        tol=0.05, n_categories=6, variables=categorical+discrete)),
    ('categorical_enc', ce.OrdinalCategoricalEncoder(
        encoding_method='ordered', variables=categorical+discrete)),

    # discretisation + encoding
    ('discretisation', dsc.EqualFrequencyDiscretiser(
        q=5, return_object=True, variables=numerical)), # Create 5 intervals
    ('encoding', ce.OrdinalCategoricalEncoder(
        encoding_method='ordered', variables=numerical)), # encode results intervals reorder to create monotonic

    # feature Scaling
    ('scaler', StandardScaler()), # scale cuz we use a linear model. Better to use RobustScaler to handle outliers
    
    # regression
    ('model', Lasso(random_state=0))
])


# let's fit the pipeline
house_pipe.fit(X_train, y_train)

# let's get the predictions
X_train_preds = house_pipe.predict(X_train)
X_test_preds = house_pipe.predict(X_test)

# check model performance:
print('train rmse: {}'.format(sqrt(mean_squared_error(y_train, X_train_preds))))
print('train r2: {}'.format(r2_score(y_train, X_train_preds)))
print()
print('test rmse: {}'.format(sqrt(mean_squared_error(y_test, X_test_preds))))
print('test r2: {}'.format(r2_score(y_test, X_test_preds)))


# let's explore the importance of the features
# the importance is given by the absolute value of the coefficient
# assigned by the Lasso
importance = pd.Series(np.abs(house_pipe.named_steps['lasso'].coef_))
importance.index = list(final_columns)+['LotFrontage_na', 'MasVnrArea_na',  'GarageYrBlt_na']
importance.sort_values(inplace=True, ascending=False)
importance.plot.bar(figsize=(18,6))
```




OLD CODE (from May 2020)
## ENCODING
FIT ONLY ON TRAIN SET cuz overfitting otherwise. Hint: fit_transform --> train. So fit on train and then transform on train,val and test
ENCODING WHEN THERE IS AN ORDER, category. Not number of


##### Get list of categorical variables
```
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)
or
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
```
##### Convert categorical variables into integers using LabelEncoder (for when there are several different names). It can already be integers but need to encode them into labels for new features
// Essential to encode categorical vars into numerical for the algorithms. Missing values treatment before
```
from sklearn.preprocessing import LabelEncoder
cat_features = ['ip', 'app', 'device', 'os', 'channel'] # Categories to encode
encoder = LabelEncoder()
for feature in cat_features: # Goal is to create new features in the same dataframe. Name format should be "feature_labels"
    encoded = encoder.fit_transform(df[feature]) # Apply the label encoder to each column
    df[feature+'_labels'] = encoded
``` 
// Then you have to one-hot encode if the features are multi-class but no if too many classes --> use LightGBM model then. 

### One-hot encoding
Strategy: Only one-hot encode columns with relatively low cardinality (few unique values). Then, high cardinality columns can be label encoded.
##### Get cardinality (number of unique entries in each column with categorical data) to check if we can do one-hot encoding
```
object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))
sorted(d.items(), key=lambda x: x[1]) # Print number of unique entries by column, in ascending order
```
##### Columns that will be one-hot encoded (cardinality < 10)
```
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]
```
##### Columns that will be label encoded
```
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))

from sklearn.preprocessing import OneHotEncoder --> PANDAS ONE-HOT IS EASIER
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)  # ignore to avoid errors when the validation data contains classes that aren't represented in the training data and sparse = False to have a np.array and not a matrix
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols/low_cardinality_cols]))  # Apply one-hot encoder to each column with categorical data
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols/low_cardinality_cols]))   # object_cols is the list of categorical vars
OH_cols_train.index = X_train.index  # One-hot encoding removed index; put it back
OH_cols_valid.index = X_valid.index
num_X_train = X_train.drop(object_cols, axis=1) # Remove categorical columns (will replace with one-hot encoding)
num_X_valid = X_valid.drop(object_cols, axis=1)
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1) # Add one-hot encoded columns to numerical features  
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
```

### Statistics encoder so fit only on train set
##### Count encoder: replaces each categorical value with the number of times it appears in the dataset. The common/important values get their own grouping
```
import category_encoders as ce
    # Create the count encoder
    count_enc = ce.CountEncoder(cols=cat_features)
    # Learn encoding from the training set
    count_enc.fit(train[cat_features])
    # Apply encoding to the train and validation sets and add suffix "_count". You can join two dataframes with the same index using .join
    train_encoded = train.join(count_enc.transform(train[cat_features]).add_suffix('_count'))
    valid_encoded = valid.join(count_enc.transform(valid[cat_features]).add_suffix('_count'))
```
### Target encoding: replaces a categorical value with the average value of the target. FIT ONLY ON TRAIN SET
```
target_encoder = ce.TargetEncoder(cols=cat_features)
target_encoder.fit(train[cat_features], train['target'])    # Fit the encoder using the categorical features and target
train_encoded = train.join(target_encoder.transform(train[cat_features]).add_suffix("_target"))   # Apply the encoder
```
// Same for Catboost. Better with lightGBM


### Add new columns for timestamp features day, hour, minute, and second so we can use time in the model
```
df['day'] = df['time'].dt.day.astype('uint8')
df['hour'] = df['time'].dt.hour.astype('uint8') #etc for minute and second. Important to convert time into categorical feature
```

### Train with lightGBM (after manual split)
```
import lightgbm as lgb
dtrain = lgb.Dataset(train[feature_cols], label=train['target'])  # feature_cols = ["x1", "x2", etc]
dvalid = lgb.Dataset(valid[feature_cols], label=valid['target'])
dtest = lgb.Dataset(test[feature_cols], label=test['target'])
param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 1000
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10)
```

### Feature engineering to create new variables
// Do interaction between categorical variables to have more information
```
interactions = df['category'] + "_" + ks['country'] --> "Poetry_GB" or "NarrativeFilm_US"  # Then label encode 
```

### Add interaction features for each pair of categorical features using itertools and encode them
```
import itertools
cat_features = ['ip', 'app', 'device', 'os', 'channel']
interactions = pd.DataFrame(index=clicks.index)  # df = clicks
for col1,col2 in itertools.combinations(cat_features, 2):    # Iterate through each pair of features, combine them into interaction features
    new_col_name = "_".join([col1,col2])  
    new_values = clicks[col1].map(str) + "_" + clicks[col2].map(str)  # join the values as strings with an underscore
    
    encoder = preprocessing.LabelEncoder()
    interactions[new_col_name] = encoder.fit_transform(new_values)
``` 
 
### FEATURE SELECTION. Two methods: univariate (for small data) and feature selection with L1 regularization (more powerful but slow)
##### Univariate feature selection: measure how strongly the target depends on a feature. ON TRAIN SET
// Use feature_selection.SelectKBest --> returns the K best features given some scoring function. Ex: F-value (f_classif) measures the linear dependency between the feature variable and the target. This means the score might underestimate the relation between a feature and the target if the relationship is nonlinear

```
from sklearn.feature_selection import SelectKBest, f_classif
feature_cols = df.columns.drop('target')   # Keep only features columns
selector = SelectKBest(f_classif, k=5)  # Keep 5 features and use F-value
X_new = selector.fit_transform(train[feature_cols], train['target'])  # ONLY ON TRAIN SET
```
// At this stage, we have an array with 5 features but it's not clear which ones we've kept. So we need to inverse_transform

#### L1 regularization (Lasso)
```
from sklearn.linear_model import LogisticRegression (classification), Lasso (regression pb)
from sklearn.feature_selection import SelectFromModel
X, y = train[train.columns.drop("target")], train['target']
logistic = LogisticRegression(C=1, penalty="l1", random_state=7).fit(X, y)  # Set the regularization parameter C=1
model = SelectFromModel(logistic, prefit=True)
X_new = model.transform(X)
```

##### Get back the features we've kept and set zero for all other features
```
selected_features = pd.DataFrame(selector/model.inverse_transform(X_new), 
                                 index=train.index, 
                                 columns=feature_cols)
```

##### Dropped columns have values of all 0s, so var is 0, drop them
```
selected_columns = selected_features.columns[selected_features.var() != 0]
```
--> Function for Lasso: to give the selected features
```
def select_features_l1(X, y):
    """ Return selected features using logistic regression with an L1 penalty """
    logistic = LogisticRegression(C=0.1, penalty="l1", random_state=7).fit(X, y)  
    model = SelectFromModel(logistic, prefit=True)
    X_new = model.transform(X)
    selected_features = pd.DataFrame(model.inverse_transform(X_new), 
                                 index=X.index, 
                                 columns=X.columns)
    selected_columns = selected_features.columns[selected_features.var() != 0]
    return selected_columns
```  
Warning!!
Fitting a label encoder to a column in the training data creates a corresponding integer-valued label for each unique value that appears in the training data. 
In the case that the validation data contains values that don't also appear in the training data, the encoder will throw an error, because these values won't have an integer assigned to them. 
Notice that the 'Condition2' column in the validation data contains the values 'RRAn' and 'RRNn', but these don't appear in the training data -- thus, if we try to use a label encoder with scikit-learn, the code will throw an error.
--> Solution: Create a custom label encoder to deal with new categories. 
