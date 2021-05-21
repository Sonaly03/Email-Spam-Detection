
import os
import numpy as np
import nltk

from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from nltk.tokenize import word_tokenize
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing import sequence


#Funtion to get features from the training set
def features_from_train(train_dir):
    start = 0
    #Creating an empty feature matrix
    features_matrix = np.zeros((22533,3000))
    #Labels for the feature matrix
    labels = np.zeros(22533)
    #Get all emails for iterating one by one
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]
    #Iterate through each email
    for j in emails:
        with open(j, encoding="latin1") as m:
            words_in_line = []
            #Get all the words from the given line
            for line in m:
                words = line.split()
                words_in_line += words
            for word in words_in_line:
                word_in_dict = 0
                #Get the corresponding integer value from dictionary for the given word
                for i,d in enumerate(my_dictionary):
                    if d[0] == word:
                        word_in_dict = i
                        features_matrix[start,word_in_dict] = words_in_line.count(word)
        #Label that file according to the ending of the filename. .spam or .ham
        labels[start] = int(j.split(".")[-2] == 'spam')
        #Go to the next document
        start = start + 1
    return features_matrix,labels

#Function to get features from the testing set
def features_from_test(test_dir):
    start = 0
    #Creating an empty feature matrix
    features_matrix = np.zeros((11175,3000))
    #Labels for the feature matrix
    labels = np.zeros(11175)
    #Get all emails for iterating one by one
    emails = [os.path.join(test_dir,f) for f in os.listdir(test_dir)]
    #Iterate through each email
    for j in emails:
        with open(j, encoding="latin1") as m:
            words_in_line = []
            #Get all the words from the given line
            for line in m:
                words = line.split()
                words_in_line += words
            for word in words_in_line:
                word_in_dict = 0
                #Get the corresponding integer value from dictionary for the given word
                for i,d in enumerate(my_dictionary):
                    if d[0] == word:
                        word_in_dict = i
                        features_matrix[start,word_in_dict] = words_in_line.count(word)
        #Label that file according to the ending of the filename. .spam or .ham
        labels[start] = int(j.split(".")[-2] == 'spam')
        #Go to the next document
        start = start + 1
    return features_matrix,labels

#Creating the words to vector dictionary
def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]
    words_in_line = []
    #read each email line by line
    for i in emails:
        with open(i, encoding='latin1') as m:
            content = m.read()
            words_in_line += nltk.word_tokenize(content)
    #removing stopwords and lower casing everything
    dictionary = [word for word in words_in_line if word not in stopwords.words('english')]
    dictionary = [word.lower() for word in dictionary if word.isalpha()]
    dictionary = Counter(dictionary)
    dictionary = dictionary.most_common(3000)
    return dictionary

path = 'email/plaintext/quickdataset/train'
my_dictionary = make_Dictionary(path)
train_matrix,train_labels = features_from_train(path)


test_dir = 'email/plaintext/full_enron_dataset/test'
test_matrix,test_labels = features_from_test(test_dir)

max_words = 500
train_matrix = sequence.pad_sequences(train_matrix, maxlen=max_words)
test_matrix = sequence.pad_sequences(test_matrix, maxlen=max_words)


embedding_size=32
model=Sequential()
model.add(Embedding(63090, embedding_size, input_length=500))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
batch_size = 64
num_epochs = 1
X_valid, y_valid = train_matrix[:batch_size], train_labels[:batch_size]
X_train2, y_train2 = train_matrix[batch_size:], train_labels[batch_size:]
model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)

scores = model.evaluate(test_matrix, test_labels, verbose=0)
print('Test accuracy:', scores[1])
