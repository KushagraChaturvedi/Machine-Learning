import numpy  as np

class LinearRegressor:
    def __init__(self, weight=1, bias=1):
        self.w = weight
        self.b = bias
    
    # stochastic Gradient?
    def Gradient(self, X, Y, LR=0.0001):
        # variable init
        n = len(X)
        X = np.array(X)
        Y = np.array(Y)

        # Claculating slope of loss function with respect to Weight and Biases.
        pdw = -(2/n) * np.sum((Y - (X * self.w + self.b)) * X)    
        pdb = -(2/n) * np.sum((Y - (X * self.w + self.b)))

        # Updating weights and biases.
        self.w -= (LR * pdw) 
        self.b -= (LR * pdb)

    
    def trainModel(self, X, Y, LR=0.0001, E=100):
        for i in range(E):
            self.Gradient(X, Y, LR)

    def modelError(self, X, Y):
        n = len(X)
        X = np.array(X)
        Y = np.array(Y)        

        error = np.sum((Y - self.predict(X))**2) / n
        return error
    
    def predict(self, x):
        y = (self.w * x) + self.b
        return y

class MVRegressor():
    pass

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # dummy_train = pd.read_csv('Regressor/Dummy-data/train.csv').dropna()
    # dummy_test = pd.read_csv('Regressor/Dummy-data/test.csv')
    
    # xTrain = dummy_train['x']
    # yTrain = dummy_train['y']
    # xTest = dummy_test['x']
    # yTest = dummy_test['y']

    dummy_train = [(0, 25), (2, 45), (1, 30), (3, 52), (9, 96), (3, 61), (2, 75), (8, 89), (6, 71), (7, 64), (4, 98)]
    dummy_test = [(1, 31), (2, 43), (3, 45), (4, 75), (5, 77), (6, 81), (7, 86), (8, 85), (9, 90)]
    
    xTrain = list(map(lambda x: x[0], dummy_train))
    yTrain = list(map(lambda x: x[1], dummy_train))
    xTest = list(map(lambda x: x[0], dummy_test))
    yTest = list(map(lambda x: x[1], dummy_test))
  
    reg = LinearRegressor()

    print(f"\nBefore Training\nModel error: {reg.modelError(xTest, yTest)}\n\n")
    reg.trainModel(xTrain, yTrain, LR=0.01)
    print(f"After Traning\nModel error: {reg.modelError(xTest, yTest)}")


    plt.scatter(xTest, yTest)
    x = np.array([0, max(xTest)])
    plt.plot(x, reg.predict(x), color='red')
    plt.show()



   




   

