# 8)Assuming a set of documents that need to be classified, use the naïve Bayesian Classifier model to perform this task. Built-in Java classes/API can be used to write the program. Calculate the accuracy, precision, and recall for your data set.
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

msg = pd.read_csv('naivetext.csv', names=['message', 'label'])
print(msg)

msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

x = msg.message
y = msg.labelnum


# # # splitting the dataset into train and test data
xtrain, xtest, ytrain, ytest = train_test_split(x, y)

count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm = count_vect.transform(xtest)

# # # Training Naive Bayes (NB) classifier on training data.
clf = MultinomialNB()
clf.fit(xtrain_dtm, ytrain)
predicted = clf.predict(xtest_dtm)

# # # printing accuracy, Confusion matrix, Precision and Recall
print('\n Accuracy of the classifer is',
      accuracy_score(ytest, predicted))
print('\n Confusion matrix')
print(confusion_matrix(ytest, predicted))
print('\n The value of Precision', precision_score(ytest, predicted))
print('\n The value of Recall', recall_score(ytest, predicted))


##########VIVA##############
# The (train_test_split) function is for splitting a single dataset for two different purposes: training and testing. The training subset is for building your model. The testing subset is for using the model on unknown data to evaluate the performance of the model.
# CountVectorizer is a great tool provided by the scikit-learn library in Python. It is used to transform a given text into a vector on the basis of the frequency (count) of each word that occurs in the entire text.
