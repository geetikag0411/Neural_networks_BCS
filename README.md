# Neural_networks_BCS
Recruitment task
Neural networks comprise of node layer(containing input),one or more hidden layers(one in my case) and output layer.
![image](https://user-images.githubusercontent.com/96648258/179774169-63328fbc-d9fe-45d2-99de-fdbaff04b4c3.png)

Each node connects to another and has an associated weight and bias.

Firstly, I have created the class Layer,which generates random weights and biases.And function forward gives output corresponding to input.
![image](https://user-images.githubusercontent.com/96648258/179772386-7428371c-b864-491c-8921-5b3300facd07.png)

sigmoid is the activation func and sigmoid_prime is its derivative.

'Activation' class gives the activated output.

Class 'error' as the name suggests calculates the error using mse(mean square error) function and its derivative msePrime

output_error ![image](https://user-images.githubusercontent.com/96648258/179777312-d78b8305-8b22-4212-8d02-12ca648ef900.png)

output_error_gradient![image](https://user-images.githubusercontent.com/96648258/179777570-3139d90a-b673-4660-b58a-07db5fe30215.png)![image](https://user-images.githubusercontent.com/96648258/179777634-0936dda1-1340-4921-837e-1e882fecb094.png)

backward_propagation calulates ![image](https://user-images.githubusercontent.com/96648258/179778305-f79f3076-61b1-41f1-a9d0-e3540daf4935.png)![image](https://user-images.githubusercontent.com/96648258/179778674-9d076e87-88a6-420c-a1ca-b365dd080f75.png)

Input layer consists of 3 neurons,hidden 4 and output 3 neurons.

X is the 3 sets of input to the network

Y is the expected output

Rest of the code is self explanatory
