import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets

with open('whitewine.csv', 'r') as file:
    file.readline()
    reader = csv.reader(file, delimiter=';')
    lines = list(reader)
    result = np.array(lines).astype('float')
    x = result[:, 0:11]
    y = result[:, 11]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=np.random)

logreg = LogisticRegression()
logreg.fit(x_train, y_train)

score = logreg.score(x_test, y_test)
print(score)