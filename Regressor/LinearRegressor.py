from multiprocessing import dummy
import numpy  as np

class LinearRegressor:
    def __init__(self, weight=0, bias=0):
        self.w = weight
        self.b = bias
    
    def stochasticGradient(self, X, Y, LR=0.0001):
        # variable init
        n = len(X)
        X = np.array(X)
        Y = np.array(Y)

        # Claculating slope of loss function with respect to Weight and Biases.
        pdw = np.sum(-(2/n) * (Y - (X * self.w + self.b)) * X)    
        pdb = np.sum(-(2/n) * (Y - (X * self.w + self.b)))

        # Updating weights and biases.
        self.w -= (LR * pdw) 
        self.b -= (LR * pdb)

    
    def trainModel(self, X, Y, LR=0.0001, E=10):
        for i in range(E):
            self.stochasticGradient(X, Y, LR)

    def modelError(self, X, Y):
        n = len(X)
        X = np.array(X)
        Y = np.array(Y)        

        error = ((np.sum(Y - self.predict(X)) / n) / (np.sum(X)/n)) * 100
        return error
    
    def predict(self, x):
        y = (self.w * x) + self.b
        return y

class MVRegressor():
    pass

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    
    dummy_train = pd.read_csv('Regressor/Dummy-data/train.csv').dropna()
    dummy_test = pd.read_csv('Regressor/Dummy-data/test.csv').dropna()
    
    xTrain = dummy_train['x']
    yTrain = dummy_train['y']
    xTest = dummy_test['x']
    yTest = dummy_test['y']
  
    reg = LinearRegressor(weight=0, bias=0)

    print(f"\nBefore Training\nModel error: {reg.modelError(xTest, yTest)}\n\n")
    reg.trainModel(xTrain, yTrain, LR=0.0001)
    print(f"After Traning\nModel error: {reg.modelError(xTest, yTest)}")


    plt.scatter(xTest, yTest)
    x = np.array([0, 100])
    plt.plot(x, reg.predict(x), color='red')
    plt.show()



   

