#!/usr/bin/env python
# coding: utf-8

# # DAY-3

# # 1.SENTIMENT ANALYSIS

# In[16]:


pip install wordcloud


# In[17]:


pip install textblob


# In[18]:


#Load the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import spacy
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

import os

import warnings
warnings.filterwarnings('ignore')


# In[19]:


#importing the training data
imdb_data=pd.read_csv("C://Users//vaish//IMDB Dataset.csv")
print(imdb_data.shape)
imdb_data.head(10)


# In[20]:


#Summary of the dataset
imdb_data.describe()


# In[21]:


#sentiment count
imdb_data['sentiment'].value_counts()


# In[22]:


#sentiment count
imdb_data['sentiment'].value_counts()


# In[23]:


#split the dataset  
#train dataset
train_reviews=imdb_data.review[:40000]
train_sentiments=imdb_data.sentiment[:40000]
#test dataset
test_reviews=imdb_data.review[40000:]
test_sentiments=imdb_data.sentiment[40000:]
print(train_reviews.shape,train_sentiments.shape)
print(test_reviews.shape,test_sentiments.shape)


# In[24]:


pip install nltk


# In[25]:


pip install ToktokTokenizer


# In[18]:


pip install nltk


# In[2]:


# Importing necessary libraries
from nltk.tokenize import ToktokTokenizer
import nltk

# Tokenization of text
tokenizer = ToktokTokenizer()

# Setting English stopwords
stopword_list = nltk.corpus.stopwords.words('english')


# In[13]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Assuming norm_train_reviews is a list of strings containing positive reviews
norm_train_reviews = ["Positive review 1", "Positive review 2", "Positive review 3"]

# Combine the positive reviews into a single string
positive_text = ' '.join(norm_train_reviews)

# Word cloud for positive review words
plt.figure(figsize=(10, 10))
WC = WordCloud(width=1000, height=500, max_words=500, min_font_size=5)
positive_words = WC.generate(positive_text)

# Display the word cloud
plt.imshow(positive_words, interpolation='bilinear')
plt.axis('off')  # Turn off the axis
plt.show()


# In[16]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Assuming norm_train_reviews is a list of strings containing negative reviews
norm_train_reviews = ["Negative review 1", "Negative review 2", "Negative review 3"]

# Choose a valid index for negative reviews (e.g., 0, 1, or 2)
index_for_negative_review = 0  # Replace with the correct index

# Word cloud for negative review words
plt.figure(figsize=(10, 10))

# Check if the chosen index is within the valid range
if 0 <= index_for_negative_review < len(norm_train_reviews):
    negative_text = norm_train_reviews[index_for_negative_review]
    WC = WordCloud(width=1000, height=500, max_words=500, min_font_size=5)
    negative_words = WC.generate(negative_text)

    # Display the word cloud
    plt.imshow(negative_words, interpolation='bilinear')
    plt.axis('off')  # Turn off the axis
    plt.show()
else:
    print("Invalid index. Choose a valid index within the range of norm_train_reviews.")


# In[15]:


pip install tensorflow


# In[ ]:





# In[27]:


import nltk

# Download the stopwords resource
nltk.download('stopwords')


# In[28]:


import nltk
from nltk.tokenize import ToktokTokenizer

# Download the stopwords resource
nltk.download('stopwords')

# Tokenization of text
tokenizer = ToktokTokenizer()

# Setting English stopwords
stopword_list = nltk.corpus.stopwords.words('english')


# In[29]:


import pandas as pd

# Assuming 'imdb_data.csv' is your dataset file
# Adjust the file path accordingly
imdb_data = pd.read_csv("C://Users//vaish//IMDB Dataset.csv")

# Define your denoise_text function here
def denoise_text(text):
    # Your denoising logic goes here
    return text

# Apply the function to the 'review' column
imdb_data['review'] = imdb_data['review'].apply(denoise_text)


# In[30]:


pip install re


# In[31]:


import re
import pandas as pd

# Assuming 'imdb_data.csv' is your dataset file
# Adjust the file path accordingly
imdb_data = pd.read_csv("C://Users//vaish//IMDB Dataset.csv")

# Define your remove_special_characters function here
def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', text)
    return text

# Apply the function to the 'review' column
imdb_data['review'] = imdb_data['review'].apply(remove_special_characters)


# In[32]:


import pandas as pd

# Load or define your dataset
# Assuming 'imdb_data.csv' is your dataset file
# Adjust the file path accordingly
imdb_data = pd.read_csv("C://Users//vaish//IMDB Dataset.csv")

# Define your simple_stemmer function here
def simple_stemmer(text):
    # Your simple stemming logic goes here
    return text

# Apply the function to the 'review' column
imdb_data['review'] = imdb_data['review'].apply(simple_stemmer)


# In[33]:


import nltk
nltk.download('stopwords')


# In[34]:


import nltk
nltk.data.path.append("path/to/nltk_data")


# In[35]:


#normalized train reviews
norm_train_reviews=imdb_data.review[:40000]
norm_train_reviews[0]
#convert dataframe to string
#norm_train_string=norm_train_reviews.to_string()
#Spelling correction using Textblob
#norm_train_spelling=TextBlob(norm_train_string)
#norm_train_spelling.correct()
#Tokenization using Textblob
#norm_train_words=norm_train_spelling.words
#norm_train_words


# In[36]:


#normalized train reviews
norm_train_reviews=imdb_data.review[:40000]
norm_train_reviews[0]
#convert dataframe to string
#norm_train_string=norm_train_reviews.to_string()
#Spelling correction using Textblob
#norm_train_spelling=TextBlob(norm_train_string)
#norm_train_spelling.correct()
#Tokenization using Textblob
#norm_train_words=norm_train_spelling.words
#norm_train_words


# In[37]:


#Normalized test reviews
norm_test_reviews=imdb_data.review[40000:]
norm_test_reviews[45005]
##convert dataframe to string
#norm_test_string=norm_test_reviews.to_string()
#spelling correction using Textblob
#norm_test_spelling=TextBlob(norm_test_string)
#print(norm_test_spelling.correct())
#Tokenization using Textblob
#norm_test_words=norm_test_spelling.words
#norm_test_words


# In[38]:


pip install scikit-learn


# In[39]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Assuming norm_train_reviews is your preprocessed text data
# Adjust the data accordingly
tv = TfidfVectorizer(min_df=1, max_df=1, use_idf=True, ngram_range=(1, 3))

# Transformed train reviews
tv_train_reviews = tv.fit_transform(norm_train_reviews)


# In[ ]:


pip install scikit-learn


# In[3]:


import pandas as pd
from sklearn.preprocessing import LabelBinarizer

# Assuming you have a code cell where you load or define imdb_data
# Replace this example with your actual code to load or define the DataFrame
imdb_data = pd.read_csv("C://Users//vaish//IMDB Dataset.csv")

# Now you can use the LabelBinarizer code
lb = LabelBinarizer()
sentiment_data = lb.fit_transform(imdb_data['sentiment'])
print(sentiment_data.shape)


# In[4]:


#Spliting the sentiment data
train_sentiments=sentiment_data[:40000]
test_sentiments=sentiment_data[40000:]
print(train_sentiments)
print(test_sentiments)


# In[7]:


from sklearn.feature_extraction.text import CountVectorizer

# Example data for illustration
cv_train_reviews = [
    "This movie is great!",
    "I enjoyed watching it.",
    "The plot is confusing.",
    "The acting is superb."
]

train_sentiments = [1, 1, 0, 1]  # 1 for positive sentiment, 0 for negative sentiment

# Check the data types and contents of cv_train_reviews
print("Type of cv_train_reviews:", type(cv_train_reviews))
print("Contents of cv_train_reviews:", cv_train_reviews)

# Check the data types and contents of train_sentiments
print("\nType of train_sentiments:", type(train_sentiments))
print("Contents of train_sentiments:", train_sentiments)

# Text representation using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(cv_train_reviews)

# Display the feature names and the transformed data
print("\nFeature names:", vectorizer.get_feature_names_out())
print("Transformed data (sparse matrix):\n", X.toarray())


# In[ ]:





# In[ ]:


# Assuming you have the remove_special_characters function defined
def remove_special_characters(text):
    # Your logic to remove special characters
    return text

# Assuming you have the denoise_text function defined
def denoise_text(text):
    # Your denoising logic here
    return text

# Load the test data from the CSV file into a DataFrame
imdb_test_data = pd.read_csv("C://Users//vaish//IMDB Dataset.csv")

# Assuming you have a variable named imdb_test_data with the test reviews
# Apply the same preprocessing steps as you did for the training data
processed_test_reviews = imdb_test_data['review'].apply(denoise_text)
processed_test_reviews = processed_test_reviews.apply(remove_special_characters)
# Add any other preprocessing steps as needed

# Transform test reviews using the CountVectorizer or TfidfVectorizer
cv_test_reviews = cv.transform(processed_test_reviews)  # Use cv for CountVectorizer or tv for TfidfVectorizer

# Predict using the trained logistic regression model
lr_bow_predict = lr.predict(cv_test_reviews)

print(lr_bow_predict)


# In[ ]:





# In[13]:


pip install scikit-learn


# In[ ]:


# Classification report for bag of words
mnb_bow_report = classification_report(test_sentiments, mnb_bow_predict, target_names=['Positive', 'Negative'])
print(mnb_bow_report)

# Classification report for TF-IDF features
mnb_tfidf_report = classification_report(test_sentiments, mnb_tfidf_predict, target_names=['Positive', 'Negative'])
print(mnb_tfidf_report)


# In[22]:


from sklearn.linear_model import LogisticRegression

def train_logistic_regression(train_data, train_labels):
    # Assuming you have a CountVectorizer (cv) defined and trained during training
    # Also, assuming that train_data is already transformed using the same CountVectorizer

    # Train the logistic regression model
    lr = LogisticRegression()
    lr.fit(train_data, train_labels)

    # Return the trained model
    return lr


# In[ ]:


from sklearn.linear_model import SGDClassifier

# Training the linear SVM
svm = SGDClassifier(loss='hinge', max_iter=500, random_state=42)

# Fitting the SVM for bag of words
svm_bow = svm.fit(cv_train_reviews, train_sentiments)


# In[ ]:


#Predicting the model for bag of words
svm_bow_predict=svm.predict(cv_test_reviews)
print(svm_bow_predict)
#Predicting the model for tfidf features
svm_tfidf_predict=svm.predict(tv_test_reviews)
print(svm_tfidf_predict)


# In[ ]:


#Accuracy score for bag of words
svm_bow_score=accuracy_score(test_sentiments,svm_bow_predict)
print("svm_bow_score :",svm_bow_score)
#Accuracy score for tfidf features
svm_tfidf_score=accuracy_score(test_sentiments,svm_tfidf_predict)
print("svm_tfidf_score :",svm_tfidf_score)


# In[ ]:


#Classification report for bag of words 
svm_bow_report=classification_report(test_sentiments,svm_bow_predict,target_names=['Positive','Negative'])
print(svm_bow_report)
#Classification report for tfidf features
svm_tfidf_report=classification_report(test_sentiments,svm_tfidf_predict,target_names=['Positive','Negative'])
print(svm_tfidf_report)


# In[ ]:


#confusion matrix for bag of words
cm_bow=confusion_matrix(test_sentiments,svm_bow_predict,labels=[1,0])
print(cm_bow)
#confusion matrix for tfidf features
cm_tfidf=confusion_matrix(test_sentiments,svm_tfidf_predict,labels=[1,0])
print(cm_tfidf)


# In[ ]:


#training the model
mnb=MultinomialNB()
#fitting the svm for bag of words
mnb_bow=mnb.fit(cv_train_reviews,train_sentiments)
print(mnb_bow)
#fitting the svm for tfidf features
mnb_tfidf=mnb.fit(tv_train_reviews,train_sentiments)
print(mnb_tfidf)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB

# training the model
mnb = MultinomialNB()

# fitting the Multinomial Naive Bayes for bag of words
mnb_bow = mnb.fit(cv_train_reviews, train_sentiments)


# In[ ]:


#Predicting the model for bag of words
mnb_bow_predict=mnb.predict(cv_test_reviews)
print(mnb_bow_predict)
#Predicting the model for tfidf features
mnb_tfidf_predict=mnb.predict(tv_test_reviews)
print(mnb_tfidf_predict)


# In[ ]:


#Accuracy score for bag of words
mnb_bow_score=accuracy_score(test_sentiments,mnb_bow_predict)
print("mnb_bow_score :",mnb_bow_score)
#Accuracy score for tfidf features
mnb_tfidf_score=accuracy_score(test_sentiments,mnb_tfidf_predict)
print("mnb_tfidf_score :",mnb_tfidf_score)


# In[ ]:


#Classification report for bag of words 
mnb_bow_report=classification_report(test_sentiments,mnb_bow_predict,target_names=['Positive','Negative'])
print(mnb_bow_report)
#Classification report for tfidf features
mnb_tfidf_report=classification_report(test_sentiments,mnb_tfidf_predict,target_names=['Positive','Negative'])
print(mnb_tfidf_report)


# In[ ]:


#confusion matrix for bag of words
cm_bow=confusion_matrix(test_sentiments,mnb_bow_predict,labels=[1,0])
print(cm_bow)
#confusion matrix for tfidf features
cm_tfidf=confusion_matrix(test_sentiments,mnb_tfidf_predict,labels=[1,0])
print(cm_tfidf)


# In[25]:


# Word cloud for positive review words
plt.figure(figsize=(10, 10))
positive_text = norm_train_reviews[1]
WC = WordCloud(width=1000, height=500, max_words=500, min_font_size=5)
positive_words = WC.generate(positive_text)
plt.imshow(positive_words, interpolation='bilinear')
plt.show()


# In[24]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Assuming norm_train_reviews is a list of strings containing negative reviews
norm_train_reviews = ["Negative review 1", "Negative review 2", "Negative review 3"]

# Choose the appropriate index for negative reviews
index_for_negative_review = 0  # Replace with the correct index

# Word cloud for negative review words
plt.figure(figsize=(10, 10))

# Check if the chosen index is within the valid range
if 0 <= index_for_negative_review < len(norm_train_reviews):
    negative_text = norm_train_reviews[index_for_negative_review]
    WC = WordCloud(width=1000, height=500, max_words=500, min_font_size=5)
    negative_words = WC.generate(negative_text)

    # Display the word cloud
    plt.imshow(negative_words, interpolation='bilinear')
    plt.axis('off')  # Turn off the axis
    plt.show()
else:
    print("Invalid index. Choose a valid index within the range of norm_train_reviews.")


# # 2Text classification using NLP 20 news groups data set 

# In[41]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output


# Any results you write to the current directory are saved as output.


# In[43]:


# Import necessary libraries
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(newsgroups.data, newsgroups.target, test_size=0.2, random_state=42)

# Convert text data to feature vectors using TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = nb_classifier.predict(X_test_tfidf)

# Evaluate the performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test,y_pred))


# # Named Entity Recognition(NER)

# In[44]:


pip install spacy


# In[8]:


pip install --upgrade spacy


# In[6]:


# Function to read CoNLL 2003 dataset
def read_conll_data(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        sentence = []
        for line in file:
            line = line.strip()
            if line == '':
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                tokens = line.split()
                word, pos, chunk, label = tokens[0], tokens[1], tokens[2], tokens[3]
                sentence.append({'word': word, 'pos': pos, 'chunk': chunk, 'label': label})
    return sentences

# Example usage
conll_file_path = "C://Users//vaish//archive (1)//train.txt"  # Replace with the actual path to your CoNLL 2003 dataset file
dataset = read_conll_data(conll_file_path)

# Displaying the first few sentences
for i in range(3):
    print(dataset[i])
    print()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




