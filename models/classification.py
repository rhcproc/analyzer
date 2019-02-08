
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

class ClassificationHandler:
    def __init__(self, X, y):
        a,b,c,d = train_test_split(X,y)
        self.train_X = a
        self.test_X = b
        self.train_y = c
        self.test_y = d
        self.model = None
        self.model_list = [self.setLogisticRegression, 
                           self.setGaussianNB,
                           self.setKNeighborsClassifier,
                           self.setDecisionTreeClassifier,
                           self.setSVC ]

    def setLogisticRegression(self):
        print(" [ LogisticRegression ]")
        self.model = LogisticRegression()

    def setGaussianNB(self):
        print(" [ GaussianNB ]")
        self.model = GaussianNB()

    def setKNeighborsClassifier(self):
        print(" [ KNeighborsClassifier ]")
        self.model = KNeighborsClassifier()

    def setDecisionTreeClassifier(self):
        print(" [ DecisionTreeClassifier ]")
        self.model = DecisionTreeClassifier()

    def setSVC(self):
        print(" [ SVC ]")
        self.model = SVC()

    def run(self):
        for model in self.model_list:
            model()
            self.model.fit(self.train_X, self.train_y)
            print(metrics.classification_report(self.test_y, self.model.predict(self.test_X)))
            
