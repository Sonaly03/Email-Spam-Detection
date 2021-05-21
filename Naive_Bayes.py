import os
import numpy as np
import nltk

from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC

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

train_dir = 'email/plaintext/quickdataset/train'
my_dictionary = make_Dictionary(train_dir)
train_data,train_labels = features_from_train(train_dir)
test_dir = 'email/plaintext/quickdataset/test'
test_data,test_labels = features_from_test(test_dir)
model1 = Linear_SVC()
model1.fit(train_data,train_labels)
result1 = model1.predict(test_data)
print("Results for SVM Classification : ")
metrics = confusion_matrix(test_labels,result1)
print("Accuracy : " + str((metrics[0][0]+metrics[1][1])/(metrics[0][0]+metrics[1][0]+metrics[1][1]+metrics[0][1])*100) + "%")
print("Specificity : " + str(metrics[1][1]/(metrics[1][0]+metrics[1][1])*100) + "%")
print("Recall : " + str(metrics[0][0]/(metrics[0][0]+metrics[1][0])*100) + "%")
