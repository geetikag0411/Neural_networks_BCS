import numpy as np
# activation function and its derivative
def sigmoid(x):
        s = 1/(1+np.exp(-x))
        return s

def sigmoid_prime(x):
        s = s =1/(1+np.exp(-x))
        ds = s*(1-s) 
        return ds 

# loss function and its derivative
def mse(y_actual, y_expected):
    return np.mean(np.power(y_actual-y_expected, 2))

def msePrime(y_actual, y_expected):
    return 2*(y_actual-y_expected)/y_actual.size

class Layer:
    def __init__(self, no_of_inputs,no_of_neurons) :
        self.weights = 0.10 * np.random.randn(no_of_inputs,no_of_neurons)
        self.biases = np.zeros((1, no_of_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation:
    def __init__(self, inputs):
       self.forward=sigmoid(inputs)
       self.dforward=sigmoid_prime(inputs)
class error:
    def __init__(self,output,expected_output):
        self.output_error=mse(output,expected_output)
        self.output_error_gradient=msePrime(output,expected_output)
    def backward_propagation(self,dforward):
        return np.dot(dforward , self.output_error_gradient)

layer1 = Layer(3,4)
layer2 = Layer(4,3)
X = [[1,2,3],[-1,-2,2],[4,-1,2]]
layer1.forward(X)
activation1=Activation(layer1.output)
final_output1 = activation1.forward
layer2.forward(final_output1)
activation2=Activation(layer2.output)
final_output2 =activation2.forward
print(final_output2)
print("\n")
err=[]
Y=[[1,0,0],[0,1,0],[0,0,1]]
for i in range(0,3):
    err.append(error(final_output2[i],Y[i]))
    print(err[i].output_error)
    print(err[i].output_error_gradient)
    print(err[i].backward_propagation(activation2.dforward))
    print("\n")