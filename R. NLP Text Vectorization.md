# Text vectorization

Need to convert string features into **numerical features**
<br>Different methods:
- Bag of words
- TF-IDF
- Word2Vec

### Bag of words
It is basic model used in natural language processing. It's called bag of words because **the position of the word doesn't count**.
<br>It only tells whether the word is present in the document or not
<br>Basically convert word into vector
<br>There are different approaches:
- Unigram ==> Consider only one word at a time
- Bigram ==> Using two words at a time. Ex: There used, Used to, to be, be Stone, Stone age
- Trigram ==> Using three words at a time. Ex: there used to, used to be ,to be Stone
- Ngram(using n words at a time)

Use ```CountVectorizer```from ```sklearn``` for BoW (here we take into account bigram and remove stopwords)
```
doc1 = 'Game of Thrones is an amazing tv series!'
doc2 = 'Game of Thrones is the best tv series!'
doc3 = 'Game of Thrones is so great'

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english',ngram_range=(2,2))
X = vectorizer.fit_transform([doc1,doc2,doc3])
df_bow_sklearn = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())
df_bow_sklearn.head()
```

### TF-IDF
TF-IDF is basically the same than BoW except that it computes a **score importance** rather than frequency of a word.
<br>TF-IDF stands for Term Frequency-Inverse Document Frequency which basically tells the importance of the word in the corpus or dataset. 
<br>TF-IDF contains two concepts: *Term Frequency(TF)* and *Inverse Document Frequency(IDF)*

##### Term Frequency 
How frequently the word appears in the document or corpus. 
<br>Term frequency can be defined as: **TF = Nb of time word appears in the document / Nb of words in the document**

##### Inverse Document Frequency 
Finding out importance of the word
<br>It is based on the fact that less frequent words are more informative and important
<br>Inverse Document Frequency can be defined as: **IDF = Nb of documents / Nb of documents in which the word appears*

**TF-IDF is basically a multiplication between TF table and IDF table**
<br>It reduces values of common word that are used in different document
<br>We can apply **log** to formula to reduce the value of frequency count

BoW and TF-IDF codes are similar
```
Count_vec = CountVectorizer(tokenizer=tokenize, ngram_range=(1,2), min_df=2, stop_words=stopwords)          #Bag Of Words
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, ngram_range=(1,2), min_df=2, stop_words=stopwords)   #TF IDF

#Transform list of documents in a sklearn matrix to make predictions
BoW_matrix = Count_vec.fit_transform(X)   
tfidf_matrix = tfidf_vectorizer.fit_transform(X) 

# We can add the different information (POS tag, etc) in these matrices to improve modelization

#Split train-test sets
X_BoW_train, X_BoW_test, Y_BoW_train, Y_BoW_test = train_test_split(BoW_matrix, Y, test_size=0.2, random_state = 42)
X_TFIDF_train, X_TFIDF_test, Y_TFIDF_train, Y_TFIDF_test = train_test_split(tfidf_matrix, Y, test_size=0.2, random_state = 42)
```
- tokenizer ==> represents the tokenization function that we previously created. We only need to give raw text/words as input of the Vectorizer
- ngram_range ==> if we want unigram, bigram etc
- min_df : filters on words that appear at least twice in the text. We can also use ```max_df``` for too frequent words
- stop_words ==> we can custom list of stop_words


### Google Word2Vec
**Deep learning technique with two-layer neural network**
<br>Google Word2vec takes input from large data and convert into vector space.
<br>Word2vec basically **place the word in the feature space is such as their location is determined by their meaning** i.e. words having similar meaning are clustered together and the distance between two words also have same meaning.
<br>TF-IDF contains two concepts: *Term Frequency(TF)* and *Inverse Document Frequency(IDF)*
