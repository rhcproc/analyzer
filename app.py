
from models.classification import ClassificationHandler
from sklearn import datasets

class MainHandler:
    def __init__(self):
        pass

    def load_data(self):
        X = []
        y = []
        dataset = datasets.load_iris()
        X = dataset.data
        y = dataset.target
        #print (y)
        return X,y

    def run(self):
        X,y  = self.load_data()
        CH = ClassificationHandler(X,y)
        CH.run() 

if __name__ == "__main__":
    MH = MainHandler()
    MH.run()

