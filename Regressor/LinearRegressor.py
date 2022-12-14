class Regressor:
    def __init__(self, weight=0, bias=0):
        self.w = weight
        self.b = bias
    
    def gradientDecent(self, X, Y, LR=0.01):
        n = len(X)
        pdw = 0
        pdb = 0
        for i in range(n):
            x = X[i]
            y = Y[i]
            pdb += -(2/n) * (y - (self.w * x) + self.b)
            pdw += -(2/n) * (y - (self.w * x) + self.b) * x
            # print(pdw, pdb)
        self.w -= (LR * pdw) 
        self.b -= (LR * pdb)

    def trainModel(self, X, Y, batchSize = 2, LR = 0.001):
        n = len(X)
        for i in range(0, n, batchSize):
            self.gradientDecent(X[i:i+batchSize], Y[i:i+batchSize], LR)
    
    def predict(self, x):
        y = (self.w * x) + self.b
        return y


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    # Dummy Data
    train = pd.read_csv("Dummy-data/train.csv")
    train.dropna(inplace=True)
    Y = train['y']
    X = train['x']
    
    # X = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Y = [2, 4, 6, 8, 10, 12, 14, 16, 18]
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape)
    reg = Regressor(weight=0, bias=0)

    # For small data set train it multiple times on the same data for good fit!
    for i in range(2):
        reg.trainModel(X, Y, batchSize= 32, LR = 0.0001)

    # Ploting
    plt.scatter(X, Y)
    plt.plot(list(range(0, 100)), [reg.w * i + reg.b for i in range(0, 100)], color="red") 

    plt.show()