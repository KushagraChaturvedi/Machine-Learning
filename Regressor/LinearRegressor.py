class Regressor:
    def __init__(self, weight=0, bias=0):
        self.w = weight
        self.b = bias
    
    def trainModel(self, X, Y, LR=0.01):
        n = len(X)
        pdw = 0
        pdb = 0
        for i in range(n):
            x = X.iloc[i]
            y = Y.iloc[i]
            pdb += -(2/n) * (y - (self.w * x) + self.b)
            pdw += -(2/n) * (y - (self.w * x) + self.b) * x
            # print(pdw, pdb)
        self.w -= (LR * pdw) 
        self.b -= (LR * pdb)
        return 0 

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
    

    # Normalize Data
    # sy = sum(Y)
    # Y = list(map(lambda x: x/sy, Y))

    # X = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Y = [2, 4, 6, 8, 10, 12, 14, 16, 18]
    
    # Main
    reg = Regressor(weight=0, bias=0)
    print(f"Weight: {reg.w} Bias: {reg.b}")

    epochs = 1000
    for i in range(epochs):
        reg.trainModel(X, Y, LR=.0001)
        print(f"Weight: {reg.w} Bias: {reg.b}")
    
    slope = reg.w
    c = reg.b

    q = 40
    print(f"prediction for X:{q} is {reg.predict(q)}")

    # Ploting
    plt.scatter(X, Y)
    plt.plot(list(range(0, 100)), [reg.w * i + reg.b for i in range(0, 100)], color="red") 

    plt.show()