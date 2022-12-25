import numpy as np

class Activation_ReLU:
    def forward(y):
        output = np.maximum(0, y)
        return output

class Activation_SoftMax:
    def forward(y):
        output = np.exp(y)
        output = output/sum(output)
        return output
class Activation_tanh:
    def forward(y):
        output = np.tanh(y)
        return output

class Activation_Sigmoid:
    def forward(y):
        output = np.exp(-y)
        output = 1 / 1 + output
        return output

        
if __name__ == '__main__':
    A = Activation_ReLU
    var = A.forward([1, 2, 3])
    print('run successfully', var)