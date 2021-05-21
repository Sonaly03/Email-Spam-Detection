
import os
import numpy as np
import nltk
import random

from nltk.corpus import stopwords
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold

#Funtion to get features from the training set
def features_from_train(emails):
    start = 0
    #Creating an empty feature matrix
    features_matrix = np.zeros((22533,3000))
    #Labels for the feature matrix
    labels = np.zeros(22533)
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
def features_from_test(emails):
    start = 0
    #Creating an empty feature matrix
    features_matrix = np.zeros((11175,3000))
    #Labels for the feature matrix
    labels = np.zeros(11175)
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
def make_Dictionary(emails):
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

#Path for performing cross validation
path = 'email/plaintext/quickdataset/train'
emails = [os.path.join(path,f) for f in os.listdir(path)]
#5-fold cross validation
non_training_data = cross_validation_split(emails,5)
sum=0
#Running for 5-fold cross validation
for i in range(len(non_training_data)):
    training_data = [x for x in emails if x not in non_training_data[i]]
    my_dictionary = make_Dictionary(training_data)
    train_matrix,train_labels = features_from_train(training_data)
    model1 = LinearSVC()
    model1.fit(train_matrix,train_labels)
    test_matrix,test_labels = features_from_test(non_training_data[i])
    result1 = model1.predict(test_matrix)
    print("Results for SVM Classification : ")
    metrics = confusion_matrix(test_labels,result1)
    sum=sum+(metrics[0][0]+metrics[1][1])/(metrics[0][0]+metrics[1][0]+metrics[1][1]+metrics[0][1])*100
    print("Accuracy : " + str((metrics[0][0]+metrics[1][1])/(metrics[0][0]+metrics[1][0]+metrics[1][1]+metrics[0][1])*100) + "%")

print("Avg accuracy after cross validation : "+str(sum/5)+"%")
