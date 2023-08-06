

import pandas as pd
from google.colab import files
import nltk
from nltk.corpus import stopwords
import string
uploded = files.upload()

nltk.download('stopwords')

def process_text(text):
   # to remove punctuations
   # to remove stopwords
   # return a list of clean text words

    nopunc=[char for char in text if char not in string.punctuation]
    nopunc=''.join(nopunc)

    clean_words = [word for word in nopunc.split() if word.lower not in stopwords.words('english')]


    return clean_words

df_new['text'].head().apply(process_text)

# to show  a list of tokens

message1 = 'hello oswin alex oswin oswin oswin abc world hello '
message2 ='tese test one hello'

from sklearn.feature_extraction.text import CountVectorizer
bag_of_words = CountVectorizer(analyzer=process_text).fit_transform([[message1],[message2]])
print(bag_of_words)
print()

print(bag_of_words.shape)

# converting text into tokens
from sklearn.feature_extraction.text import CountVectorizer
messages_bow = CountVectorizer(analyzer = process_text).fit_transform(df_new['text'])

# spliting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test= train_test_split(messages_bow,df_new['spam'], test_size = 0.20,random_state = 0)

# create and train naive bayes Classifier

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit(X_train,y_train)

#printing predictions
print(classifier.predict(X_train))#target values
print()
print(y_train.values)#

# evaluting the model on training
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
pred = classifier.predict(X_train)
print(classification_report(y_train,pred))
print()
print('confusion Matrix\n',confusion_matrix(y_train,pred))
print('accuracy',accuracy_score(y_train,pred))

#testing
print(classifier.predict(X_test))
print(y_test.values)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
pred = classifier.predict(X_test)
print(classification_report(y_test,pred))
print()
print('confusion Matrix\n',confusion_matrix(y_test,pred))
print('accuracy',accuracy_score(y_test,pred))

y_pred = classifier.predict(X_test)

# Print the predictions
for pred in y_pred:
    print(pred)

