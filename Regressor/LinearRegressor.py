class Regressor:
    def __init__(self, weight=0, bias=0):
        self.w = weight
        self.b = bias
    
    def trainModel(self,X, Y, LR=0.01):
        n = len(X)
        bSum = 0
        wSum = 0
        errorSum = 0
        for i in range(n):
            bSum +=  Y[i] - ((X[i] * self.w) + self.b)
            wSum += X[i] * bSum
            errorSum += bSum**2
        meanError = errorSum/n
        pdw = (2/n) * wSum
        pdb = (2/n) * bSum
        print(pdw, pdb)
        self.w -= (LR * pdw) 
        self.b -= (LR * pdb)
        return 0 
    def predict(self, x):
        y = (self.w * x) + self.b
        return y


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    X = list(np.random.randint(1, 10, size=10))
    Y = list(np.random.randint(35, 60, size=10))
    sy = sum(Y)
    Y = list(map(lambda x: x/sy, Y))
    # X = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Y = [2, 4, 6, 8, 10, 12, 14, 16, 18]
    reg = Regressor(weight=1, bias=1)
    print(f"Weight: {reg.w} Bias: {reg.b}")
    reg.trainModel(X, Y, LR=.01)
    print("training...")
    print(f"Weight: {reg.w} Bias: {reg.b}")
    print(reg.predict(5))
    c = reg.b
    slope = reg.w
    fig, ax = plt.subplots()
    ax.scatter(X, Y)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = c, c + slope*(x_max-x_min)
    ax.plot([x_min, x_max], [y_min, y_max])
    ax.set_xlim([x_min, x_max])
    plt.show()