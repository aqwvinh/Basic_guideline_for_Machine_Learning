# Tips for NLP projets

Simple pratical steps:

- Duplicates and relabel target if necessary
- Missing values
- Data cleaning
- Data vectorization (TF-IDF, Word2Vec and GloVe) and subsequent modeling


### Duplicates and relabel target
Show duplicates that have different labels
```
df_mislabeled = train.groupby(['text']).nunique().sort_values(by='target', ascending=False)
df_mislabeled = df_mislabeled[df_mislabeled['target'] > 1]['target']
df_mislabeled.index.tolist()
```

### Correct label
```
train['target_relabeled'] = train['target'].copy() 
train.loc[train['text'] == '1st duplicate text', 'target_relabeled'] = 0  # 0 is the example, choose manually for each duplicate
```

### Missing values
Create a combined list to treat missing values once
```
combine = [train, test]
```
Replace missing values in col1 and col2 with "Unknown" when it's appropriate
```
for set in combine:
  set[["col1", "col2"]] = set[["col1", "col2"]].fillna("Unknown")
```

### Data cleaning
Common functions to clean text: **lowercase, URL, HTML, emoji, punctuation, twitter handles**
To keep the train and test separation
```
ntrain = train.shape[0]
ntest = test.shape[0]
ntrain, ntest
```

Create a concatenated df to clean data once
```
df = pd.concat([train,test])
```

Lowercase
```
df['text']=df['text'].apply(lambda x : x.lower())  #import string if needed
```

Remove accent
```
pip install unidecode
import unidecode
df['text'].apply(lambda x: unidecode.unidecode(x))
```

Remove URL
```
import re
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

df['text']=df['text'].apply(lambda x : remove_URL(x))
```

Remove HTML Tags
```
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)
```

Remove multiple characters
```
def multiple_replace(dict, text):
  # Create a regular expression from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda x: dict[x.string[x.start():x.end()]], text) 
  
text = "X is THE goal of Y"
dict = {
"X" : "Mandanda",
"THE" : "the",
"Y" : "Marseille",
} 
print(multiple_replace(dict, text))
```

Remove emoji
```
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
```

Remove punctuation
```
import string
def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)
```
    
Remove twitter handles (@user)
```
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)  
    return input_txt 
```
    
Correct spelling (super long to run)
```
!pip install pyspellchecker
from spellchecker import SpellChecker
spell = SpellChecker()
def correct_spellings(text):
    corrected_text = []
    misspelled_words = spell.unknown(text.split())
    for word in text.split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)
```

#### Warning !
Resplit the processed train and test sets
```
train = df[:ntrain]
test = df[ntrain:]
train.shape, test.shape  # Check if the shapes are good
```

 ### DATA VECTORIZATION

#### Functions for tokenization and lemmatization then TF-IDF
```
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
nltk.download("punkt")
nltk.download("stopwords")
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def get_wordnet_pos(pos_tag):
    output = np.asarray(pos_tag)
    for i in range(len(pos_tag)):
        if pos_tag[i][1].startswith('J'):
            output[i][1] = wordnet.ADJ
        elif pos_tag[i][1].startswith('V'):
            output[i][1] = wordnet.VERB
        elif pos_tag[i][1].startswith('R'):
            output[i][1] = wordnet.ADV
        else:
            output[i][1] = wordnet.NOUN
    return output

def preprocessing_sentence(sentence):
    tokens = word_tokenize(sentence)
    tokens = [t.lower() for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    words_tagged = nltk.pos_tag(tokens)
    tags = get_wordnet_pos(words_tagged)
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = [lemmatizer.lemmatize(w_t[0], pos=w_t[1]) for w_t in tags]
    return lemmatized_sentence
```

Further preprocessing steps
```
# Preprocessing on the train set. You also have to do it on the test set in the end
df_train["tokens"] = df_train.text.apply(preprocessing_sentence) # create a new column "tokens" with the preprocessed text
test["tokens"] = test.text.apply(preprocessing_sentence)

# State clearly the X and y variables
X = df_train.tokens  # You have to gather all the text in one column
y = df_train.label

# split the dataset into training and validation datasets
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, y, test_size=0.25, random_state=0,
                                                   stratify=y) #random_state to have the same alea and strafify to have a balance in label

# label encode the target variable, after the test split
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_valid = encoder.transform(y_valid)
```

### TF-IDF 
You have to gather all the text in one column "tokens" first. For ML
```
from sklearn.feature_extraction.text import TfidfVectorizer
#Initialize vectorizer with identity analyzer
vect = TfidfVectorizer(analyzer = lambda x: x, max_features = 3000)

# Get the tfidf (format in sparse matrix) to fit the model. fit_transform for train and transform for valid
X_train = vect.fit_transform(X_train)
X_valid = vect.transform(X_valid)   # or X = vect.fit_transform(X) with small dataset then cross-validation
# Display the tfidf_1gram:
pd.DataFrame(X_train.toarray(), columns=vect.get_feature_names())
```
### Modeling
```
# Creating the models: adapt the selected models
models = [LogisticRegression(random_state= 1), svm.SVC(random_state= 1), SGDClassifier(random_state= 1), GradientBoostingClassifier(random_state= 1), RandomForestClassifier(random_state= 1),
            XGBClassifier(random_state= 1), DecisionTreeClassifier(random_state= 1), BaggingClassifier(DecisionTreeClassifier(random_state=1)), LGBMClassifier(random_state= 1), AdaBoostClassifier(random_state= 1),
          KNeighborsClassifier()                                      
             ]
names = ['Logistic Regression','Support Vector Classifier','Stochastic Gradient Descent','Gradient Boosting Tree','Random Forest',
         'XGBoost', 'Decision Tree', 'Bagging Decision Tree', 'Light GBM', 'Adaboost', "KNN"
         ]
         
# Define an error function: F1  # You can adapt the scoring. This is cross-validation (with small dataset)
from sklearn.model_selection import KFold, cross_val_score
def f1(model, X, y):
    f1 = cross_val_score(model, X, y, scoring="f1", cv=5)
    return f1

# Perform 5-folds cross-validation to evaluate the models 
for model, name in zip(models, names):
    # F1 score
    score = f1(model, X, y)
    print(f"- {name}: Mean: {round(score.mean(),3)}, Std: {round(score.std(),3)}")
    
# Predictions with the best model
best_model = SGDClassifier(alpha = 0.0001, loss = "hinge", penalty = "l2", n_iter_no_change= 100, max_iter = 500)  # You can do gridsearch to tune 
best_model.fit(X, y)

#Get tfidf for test set
X_test = vect.transform(test.token)

#Get predictions labels
y_pred = best_model.predict(X_test)
```


### WORD2VEC
Word2Vec creates a dense representation for each word, such that words appearing in similar contexts have similar vectors. 
<br>To get an embedding for the entire tweet, the mean of all vectors for the words in the tweet are taken. 
<br>The assumption now is that similar tweets have similar vectors.

Start with
```
!python -m spacy download en_core_web_lg  #Then once it is downloaded, restart the workin environment and comment this line

# Importation
# import spacy for NLP and re for regular expressions
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import re

# import sklearn transformers, models and pipelines
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV

# Load the small language model from spacy
nlp = spacy.load('en_core_web_sm')

# set pandas text output to 400
pd.options.display.max_colwidth = 400

# Load the en_core_web_lg model
nlp = spacy.load('en_core_web_lg', disable=["tagger", "parser", "ner"])

# create train set by getting the document vector
docs_train = [nlp(doc).vector for doc in train.text]
X_train = np.vstack(docs_train)
print('Shape of train set: {}'.format(X_train.shape))

# create test set likewise
docs_test = [nlp(doc).vector for doc in test.text]
X_test = np.vstack(docs_test)
print('Shape of test set: {}'.format(X_test.shape))

# create target
y_train = train.target.copy()
```


### TIPS: Create Machine Learning Pipeline
Creating a pipeline is important to have a robust workflow. 
<br>It ensures that all preprocessing steps that are learned on data are done within the cross-validation, to ensure that no data is leaked to the model.

In this case, it doesn't add much value since the only step in the pipeline is an estimator (here a logistic regression). 
<br>However, since it's useful for pipelines with data preprocessing steps that are learned on data, such standard scaling.

One advantage even when just using an estimator is that I can treat the estimator like a hyperparameter in the grid search.

#### WORD2VEC BASELINE WITH LOG REG 
```
# create machine learning pipeline
word2vec_pipe = Pipeline([('estimator', LogisticRegression())])
# cross validate
print('F1 score: {:.3f}'.format(np.mean(cross_val_score(word2vec_pipe, X_train, y_train, scoring = 'f1'))))
# fit pipeline
word2vec_pipe.fit(X_train, y_train)
# predict on test set
y_pred = word2vec_pipe.predict(X_test)
# submit prediction
sub=pd.DataFrame({'id':test['id'],'target':y_pred})
sub.to_csv('submission.csv',index=False)
print("Your submission was successfully saved!")

### WORD2VEC TUNING WITH SVC
# create a parameter grid
param_grid = [{'estimator' : [LogisticRegression()], 
               'estimator__C' : np.logspace(-3, 3, 7)},  # from -3 to 3, randomly gives 7 values
              {'estimator' : [SVC()], 
               'estimator__C' : np.logspace(-1, 1, 3), 
               'estimator__gamma' : np.logspace(-2, 2, 5) / X_train.shape[0]}]
               
# create a RandomizedSearchCV object
word2vec_grid_search = GridSearchCV(
    estimator = word2vec_pipe,
    param_grid = param_grid,
    scoring = 'f1',
    n_jobs = -1,
    refit = True,
    verbose = 1,
    return_train_score = True
)
# fit RandomizedSearchCV object
word2vec_grid_search.fit(X_train, y_train)

# print grid search results
cols = ['param_estimator',
        'param_estimator__C',    
        'param_estimator__gamma',
        'mean_test_score',
        'mean_train_score']
pd.options.display.max_colwidth = 50
word2vec_grid_search_results = pd.DataFrame(word2vec_grid_search.cv_results_).sort_values(by = 'mean_test_score', 
                                                                                          ascending = False)
word2vec_grid_search_results[cols].head(10)

# predict on test set with the best model from the randomized search
pred = word2vec_grid_search.predict(X_test)
```


### GLOVE for vectorization for LSTM. 
You have to upload to your environment the file glove.6B.100d. For DL. Long and not efficient
```
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
stop=stopwords.words('english')

def create_corpus(df):   # df = pd.concat([train,test])
    corpus=[]
    for tweet in tqdm(df['text']):
        words=[word.lower() for word in word_tokenize(tweet) if((word.isalpha()==1) & (word not in stop))]
        corpus.append(words)
    return corpus

corpus=create_corpus(df)

embedding_dict={}
with open('../glove.6B.100d.txt','r') as f:   
    for line in f:
        values=line.split()
        word=values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors
f.close()

word_index=tokenizer_obj.word_index
print('Number of unique words:',len(word_index))

num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,100))

for word,i in tqdm(word_index.items()):
    if i > num_words:
        continue
    
    emb_vec=embedding_dict.get(word)
    if emb_vec is not None:
        embedding_matrix[i]=emb_vec
```      

### Baseline Model: LSTM
```
embedding_dim = 100
model=Sequential()

embedding=Embedding(num_words,embedding_dim,embeddings_initializer=Constant(embedding_matrix),
                   input_length=MAX_LEN,trainable=False)

model.add(embedding)
model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

optimzer=Adam(learning_rate=1e-5)
model.compile(loss='binary_crossentropy',optimizer=optimzer,metrics=['accuracy'])

train_LSTM=train_pad[:train.shape[0]]
test_LSTM=train_pad[train.shape[0]:]

# Split the data
X_train,X_valid,y_train,y_valid=train_test_split(train_LSTM,train['target'].values,test_size=0.2)
print('Shape of train',X_train.shape)
print("Shape of Validation ",X_valid.shape)

# Fit on the training data
from keras.callbacks import EarlyStopping
history = model.fit(x = X_train,y = y_train,batch_size=4,epochs=15, validation_data = (X_valid, y_valid),
                    callbacks=[EarlyStopping(monitor='val_loss', patience = 3)])
                    
# Plot to monitor overfitting
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();

# Make predictions on test
y_pred=model.predict(test_LSTM)
y_pred=np.round(y_pred).astype(int).reshape(3263)
```
