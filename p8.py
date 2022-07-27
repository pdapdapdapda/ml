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


xtrain, xtest, ytrain, ytest = train_test_split(x, y)

count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm = count_vect.transform(xtest)

clf = MultinomialNB()
clf.fit(xtrain_dtm, ytrain)
predicted = clf.predict(xtest_dtm)

print('\n Accuracy of the classifer is',
      accuracy_score(ytest, predicted))
print('\n Confusion matrix')
print(confusion_matrix(ytest, predicted))
print('\n The value of Precision', precision_score(ytest, predicted))
print('\n The value of Recall', recall_score(ytest, predicted))


