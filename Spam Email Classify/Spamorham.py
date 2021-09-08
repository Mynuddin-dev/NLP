# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 12:13:34 2021

@author: Mynuddin

# Naive Bayes
https://github.com/Mynuddin-dev/Machine-Learning/blob/main/Naive%20Bayes/01-Naive%20Bayes%20Theory.ipynb

1.Here i use Stemming and Countvectorizer(BoW)

2.You can check Lemmatization and TF-IDF Vectorization

3. You can check Lemmatization and Count-Vectorization

4. You can check Stemming and TF-IDF Vectorization

or You can check GussianNB

5 . DataSet Link : https://archive.ics.uci.edu/ml/datasets/sms+spam+collection

"""


import pandas as pd

messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                           names=["label", "message"])



# Cleaning Part 

messages.shape

import nltk
import re
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

len(messages)
stopwords.words('english')


  
nltk.download('punkt')
wordnet=WordNetLemmatizer()
PS = PorterStemmer()



messagess_final = []

for i in range(len(messages)):
    messa = re.sub('[^a-zA-Z]',' ', messages["message"][i])  # Replace punctuation with space.
    messa = messa.lower()            # Lowercase
    messa_words = messa.split()      # Tokenization
    
    messa_final_words = [PS.stem(word) for word in messa_words if not word in set(stopwords.words('english'))]   # Skip or Remove Stop words and Stemming
    

    final_messa = ' '.join(messa_final_words)
    
    messagess_final.append(final_messa)


# Every cell contains a number, that represents the count of the word in that particular text.
# All words have been converted to lowercase.
# The words in columns have been arranged alphabetically





# Bag of words

from sklearn.feature_extraction.text import CountVectorizer
CV = CountVectorizer(max_features=5000)     ## Most Frequent 5k feature.
X = CV.fit_transform(messagess_final).toarray()






Y = messages["label"]
Y
Y = pd.get_dummies(Y , drop_first=True)

Y = Y.iloc[:,0].values     # you can choose other way just convert ndarry series

type(Y)
type(X)


# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)



 #Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detection_model = MultinomialNB().fit(X_train, y_train)


y_pred=spam_detection_model.predict(X_test)



from sklearn.metrics import confusion_matrix
Confusion_Matrics = confusion_matrix(y_test , y_pred)


from sklearn.metrics import accuracy_score

Accuracy = accuracy_score(y_test , y_pred)





