from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('tennisdata.csv')

x = data.iloc[::, :-1]
# print("\nThe First 5 values of train data is\n", x.head())
y = data.iloc[::, -1]
# print("\nThe first 5 values of Train output is\n", y.head())

le = LabelEncoder()
x.Outlook = le.fit_transform(x.Outlook)
x.Temperature = le.fit_transform(x.Temperature)
x.Humidity = le.fit_transform(x.Humidity)
x.Windy = le.fit_transform(x.Windy)
# print("\nNow the Train data is :\n", x.head())
y = le.fit_transform(y)
# print("\nNow the Train output is\n", y)

X_train, X_test, y_train, y_test = train_test_split(x, y)
classifier = GaussianNB()
classifier.fit(X_train, y_train)

print("Accuracy is:", accuracy_score(y_test, classifier.predict(X_test)))
