# Text vectorization

Need to convert string features into **numerical features**
<br>Different methods:
- Bag of words
- TF-IDF
- Word2Vec
- BERT

### Bag of words
It is basic model used in natural language processing. It's called bag of words because **the position of the word doesn't count**.
<br>It only tells whether the word is present in the document or not
<br>Basically convert word into vector
<br>There are different approaches:
- Unigram ==> Consider only one word at a time
- Bigram ==> Using two words at a time. Ex: There used, Used to, to be, be Stone, Stone age
- Trigram ==> Using three words at a time. Ex: there used to, used to be ,to be Stone
- Ngram(using n words at a time)

<br>**Limitation: the length of the vector equals the length of the entire vocabulary so it can explode.**

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

### TF-IDF and BM25
TF-IDF is basically the same than BoW except that it computes a **score importance** rather than frequency of a word.
<br>Attempt to give higher relevance scores to words that occur in fewer documents within the corpus. To that end, TF-IDF measures the frequency of a word in a text against its overall frequency in the corpus.

<br>Think of a document that mentions the word “oranges” with high frequency. TF-IDF will look at all the other documents in the corpus. If “oranges” occurs in many documents, then it is not a very significant term and is given a lower weighting in the TF-IDF text vector. If it occurs in just a few documents, however, it is considered a distinctive term. In that case, it helps characterize the document within the corpus and as such receives a higher value in the vector.

<br>TF-IDF stands for Term Frequency-Inverse Document Frequency which basically tells the importance of the word in the corpus or dataset. 
<br>TF-IDF contains two concepts: *Term Frequency(TF)* and *Inverse Document Frequency(IDF)*

**Limitation: it does not address the fact that, in short documents, even just a single mention of a word might mean that the term is highly relevant.**
<br>Solution: BM25 ==> improvement of TF-IDF because it takes into account the **length of the document**. 

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


### Word2Vec: Add meaning from context
**Deep learning technique with two-layer neural network**
<br>Google Word2vec takes input from large data and convert into vector space.
<br>**It's a predictive embedding model.**
<br>Word2vec basically **place the word in the feature space is such as their location is determined by their meaning** i.e. words having similar meaning are clustered together and the distance between two words also have same meaning.

There are two main implementations of Word2Vec:
- CBOW ==> Continuous BoW: the distributed representations of context (or surrounding words) are combined to predict the word in the middle. Faster 
- Skip-gram ==> the distributed representation of the input word is used to predict the context. Better job for unfrequent words.
<br>So the difference is in CBOW, we use surrounding words (..,wt-1, wt+1,..) to predict wt VS skip-gram, we use wt the predict surrounding words (..,wt-1, wt+1,..)

Distance in the high-dimensional space of Word2vec is measured by **cosine similarity**: the cosine of the angle between two vectors

**Advatanges**:
- The idea is very intuitive, which transforms the unlabled raw corpus into labeled data (by mapping the target word to its context word), and learns the representation of words in a classification task.
- The data can be fed into the model in an online way and needs little preprocessing, thus requires little memory.

**Disadvatanges**:
- Impossible to encode the meaning of longer passages/sentences
- Assume a sub-linear relationship into the vector space of words


### BERT and Transformers
BERT for “Bidirectional Encoder Representations from Transformers”, created to overcome the limitations of Word2Vec 
<br>BERT is able to produce contextualized word vectors by encoding a word’s position in the text in addition to the word itself.
<br>BERT’s success is based on its **Transformer architecture**, as well as the **vast amounts of data that it uses to learn**. During training, BERT “reads” the entire English-language Wikipedia and the BooksCorpus, a large collection of unpublished novels.

*How it works*:
- Transformers structure ==> Every output element is connected to every input element, and the weightings between them are dynamically calculated based upon their connection. (In NLP, this process is called attention.). Transformers **don't need to process data with an order** (VS RNN and CNN that require sequences of data)
- VS previous methods, BERT can read in both directions at once ==> that's why **bidirectional**
- Pre-trained on **two different taks**: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). 
- - Quickly, MLM ==> hide a word in a sentence and then have the program predict what word has been hidden (masked) based on the hidden word's context. 
- - NSP ==> have the program predict whether two given sentences have a logical, sequential connection or whether their relationship is simply random.
- D
 
 
