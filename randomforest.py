from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split#to split our data
from sklearn.svm import SVC#support vector classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

data = load_breast_cancer()

X = data.data
Y = data.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

clf = SVC(kernel='linear', C = 3)#defining classifier( kernel , soft margin)
clf.fit(x_train, y_train)#training the model

clf2 = KNeighborsClassifier(n_neighbors=3)
clf2.fit(x_train, y_train)

clf3 =DecisionTreeClassifier()
clf3.fit(x_train, y_train)

clf4 = RandomForestClassifier()
clf4.fit(x_train, y_train)

print(f'SVC:{clf.score(x_test, y_test)}')
print(f'KNN:{clf2.score(x_test, y_test)}')
print(f'DTC:{clf3.score(x_test, y_test)}')
print(f'RFC:{clf4.score(x_test, y_test)}')