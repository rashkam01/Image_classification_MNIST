This project explain basics of backpropogation, concept of hidden layers and building a Convolutional neural network model for MNIST dataset. MNIST dataset is a collection of images of handwritten digits. The project tries to classify each image into its appropriate number from 0 to 9. 

But there is a little challenge in the end.. What is the challenge ? Build MNIST Calculator 
We also check the accuracy of the model by adding random 10 digits from the dataset to obtain the sum of those 10 digits. 

For example: adding numbers 1+2+3+4+5+6+7+8+9+0 should give us 45 right? 

So, We pick random digits from the MNIST datasets which have been classified as 0,1,2 till 9 and then take the sum of those digits to check if our generated model also provides the sum as 45. 
Therefore, testing the accuracy of the model, if the sum is more than 45 then our model overfitts the data, otherwise it underfits the data. Therefore, we need improve accordingly till we get accurate answer. 

<a name="toc"/>

## Table of content 

1. [Backpropogation](#Backpropogation)
1. [Python Code for backpropogation](#Code_bp)
1. [Effects of Activation function and Hidden Layer](#effect)
1. [Code to Import Data](#imprt)
1. [Train and test split](#splitd)
1. [Build and Train the CNN model for classification](#b_model)
1. [Add hidden layers to check model accuracy](#hidden_layer) 
1. [Challenge - MNIST Caluculator code](#cal_code)
1. [Challenge - MNIST Calculator Solution](#cal_sol)
1. [Conclusion](#concl)

<a name="Backpropogation"/>


## Backpropogation
For a classification problem in supervised learning, the following illustrates a hand made graph for backpropogation. ![backpro](backpropo.PNG)
Considering it has single input e.g. x, (x is a pixel value that get fed into the neural network). The output neurons are typically the number of classes into which we classify the image. 
Loss funtion is the prediction of error of Nerual Net. Activation function defines the output of the node given a certain input to the node. Learning rate defines the step size, the rate at which the function moves to the local minima. 
Now, given that the learning rate is 0.1, activation function ReLU, a loss function of MSE (mean square error) we see the computation behind a backpropogation. 
![derivative](derivative.PNG)
![loss](loss_function.PNG)
![weigts](newweights.PNG)
That is how the backpropogation in neural network adjusts weights of each during each epoch to finally obtain the classification. 

The implemetation of nodes for a computation graph. Each node has a forward pass and a backward pass function, allowing for the evaluation and backpropagation of data. The code can be seen as follows: 

[Back to Table of Contents](#toc)

<b name="Code_bp"/>

## Python Code for backpropogation 

```python
from abc import ABC, abstractmethod
import math
import time


class Node(object):

    def __init__(self, inputs):
        self.inputs = inputs

    @abstractmethod
    def forward(self):
        ''' Feed-forward the result '''
        raise NotImplementedError("Missing forward-propagation method.")

    @abstractmethod
    def backward(self, d):
        ''' Back-propagate the error
            d is the delta of the subsequent node in the network '''
        raise NotImplementedError("Missing back-propagation method.")


class ConstantNode(Node):

    def __init__(self, value):
        self.output = value

    def forward(self):
        return self.output

    def backward(self, d):
        pass


class VariableNode(Node):

    def __init__(self, value):
        self.output = value

    def forward(self):
        return self.output

    def backward(self, d):
        self.output -= 0.1 * d # Gradient Descent


class AdditionNode(Node):

    def forward(self):
        self.output = sum([i.forward() for i in self.inputs])
        return self.output

    def backward(self, d):
        for i in self.inputs:
            i.backward(d)


class MultiplicationNode(Node):

    def forward(self):
        self.output = self.inputs[0].forward() * self.inputs[1].forward()
        return self.output

    def backward(self, d):
        self.inputs[0].backward(d * self.inputs[1].output)
        self.inputs[1].backward(d * self.inputs[0].output)


class MSENode(Node):

    def forward(self):
        self.output = 0.5 * (
            self.inputs[0].forward() - self.inputs[1].forward())**2
        return self.output

    def backward(self, d):
        self.inputs[0].backward(d * (self.inputs[0].output - self.inputs[1].output))
        self.inputs[1].backward(d * (self.inputs[1].output - self.inputs[0].output))


class SigmoidNode(Node):

    def forward(self):
        self.output = 1.0 / (1.0 + math.exp(-self.inputs[0].forward()))
        return self.output

    def backward(self, d):
        self.inputs[0].backward(d * self.output * (1.0 - self.output))

# We implement the "relu" as this is the activation function to be used, returns value if input is positive else zero
class ReLUNode(Node):

    def forward(self):
        ''' Returns 1 if x>0, else 0 '''
        self.output = max(0, self.inputs[0].forward())
        return self.output

    def backward(self, d):
        self.inputs[0].backward(d * (self.output > 0))
        
class TanhNode(object):

    def forward(self):
        raise NotImplementedError("Forward pass for tanh activation node has not been implemented yet.")

    def backward(self, d):
        raise NotImplementedError("Backward pass for tanh activation node has not been implemented yet.")

# Example graph as shown lecture slides
class SampleGraph(object):

    def __init__(self, x, y, w, b):
        ''' x: input
            y: expected output
            w: initial weight
            b: initial bias '''
        self.w = VariableNode(w)
        self.b = VariableNode(b)
        self.graph = MSENode([
            AdditionNode([
                MultiplicationNode([
                    ConstantNode(x),
                    self.w
                ]),
                MultiplicationNode([
                    self.b,
                    ConstantNode(1)
                ])
            ]),
            ConstantNode(y)
        ])

    def forward(self):
        return self.graph.forward()

    def backward(self, d):
        self.graph.backward(d)


class Neuron(Node):

    def __init__(self, inputs, weights, activation):
        ''' weights: list of initial weights, same length as inputs '''
        self.inputs = inputs
        # Initialize a weight for each input
        self.weights = [VariableNode(weight) for weight in weights]
        # Neurons normally have a bias, ignore for this assignment
        #self.bias = VariableNode(bias, "b")

        # Multiplication node for each pair of inputs and weights
        mults = [MultiplicationNode([i, w]) for i, w, in zip(self.inputs, self.weights)]
        # Neurons normally have a bias, ignore for this assignment
        #mults.append(MultiplicationNode([self.bias, ConstantNode(1)]))

        # Sum all multiplication results
        added = AdditionNode(mults)

        # Apply activation function
        if activation == 'sigmoid':
            self.graph = SigmoidNode([added])
        elif activation == 'relu':
            self.graph = ReLUNode([added])
        elif activation == 'tanh':
            self.graph = TanhNode([added])
        else:
            raise ValueError("Unknown activation function.")

    def forward(self):
        self.output = self.graph.forward()
        return self.output

    def backward(self, d):
        self.graph.backward(d)

    def set_weights(self, new_weights):
        for i in len(new_weights):
            self.weights[i].output = new_weights[i]

    def get_weights(self):
        return [weight.output for weight in self.weights]

if __name__ == '__main__':
    
    # Example network
    #sg = SampleGraph(2, 2, 2, 1)
    # Providing input, output and weights 
    x = 2
    y = 3
    w1 = 2
    w2 = 1
    w3 = 2
    w4 = 4
    w5 = 1
   
    #input and activation function for each neuron 
    n1 = Neuron([ ConstantNode(x) ], [ w1 ], 'relu')
    n2 = Neuron([ n1 ], [ w2 ], 'relu')
    n3 = Neuron([ n1 ], [ w3 ], 'relu')
    n4 = Neuron([ n2, n3 ], [ w4, w5 ], 'relu')
    sg = MSENode([ n4, ConstantNode(y) ])
    
    #prediction = sg.forward()
    prediction = sg.forward()
    print("Initial prediction is", prediction)

    
    #sg.backward(1)
    sg.backward(1)
    w1 = n1.get_weights()
    w2 = n2.get_weights()
    w3 = n3.get_weights()
    w4 = n4.get_weights()
    #print("w has new value", sg.w.output)
    #print("b has new value", sg.b.output)
    
    #printing new values after single update step 
    print("w1 has new value", w1)
    print("w2 has the new value", w2)
    print("w3 has the new value", w3)
    print("w4 and w5 has the new value", w4)
```

[Back to Table of Contents](#toc)

<a name="effect"/>

## Effects of Activation function and  Hidden Layer 

We notices that in the given case the activation funtion was sigmoid and found the following Test accuracy :

1. Test accuracy by setting the depth to 1 hidden layer: 0.8811
1. Test accuracy by setting the depth to 2 hidden layers: 0.7908
1. Test accuracy by setting the depth to 3 hidden layers: 0.2554

By changing the activation function from "sigmoid" to "relu" we see that that Test accuracy increases with each hidden layer:

1. Test accuracy by setting the depth to 1 hidden layer: 0.9273
1. Test accuracy by setting the depth to 2 hidden layers: 0.9407
1. Test accuracy by setting the depth to 3 hidden layers: 0.9468

By analysing the behaviour of sigmoid activation function we observed the following: For X values between -2 to 2 it changes the Y values significantly, therefore a small change in that region leads to a significant change in Y(output) values that makes it good for classification. However, towards the ends of the sigmoid function the Y value have smaller change(also know as vanishing gradient) which affects the derivative of sigmoid function. The more number of hidden layers cause the gradient to multipy and the value starts going to zero. Therefore, sigmoid activation function performs worst as the hidden layer increases. We considered changing the activation function to "relu" and it indeed improves the performance as the layer increases.

[Back to Table of Contents](#toc)

<a name="imprt"/> 

## Code to Import Data

Following is the python code to import MNIST dataset directly from the keras library: 

```python 
from __future__ import print_function
import numpy as np
np.random.seed(1234)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation #
from keras.optimizers import SGD
from keras.utils import np_utils


import matplotlib.pyplot as plt

batch_size = 128 #size of the piece 
nb_classes = 10
nb_epoch = 10
```

[Back to Table of Contents](#toc)

<a name="splitd"/> 

## Train and test split

We split the data into a training and testing data before building our model: 
```python 
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255 #be careful during normalization 
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
```

[Back to Table of Contents](#toc)

<a name="b_model"/>

## Build and Train the CNN model for classification

To understand the effects of hidden layer with different activation function, we first use only 1 hidden layer. 
Also, worth noticing is that we use an ReLU activation function as we saw it performs better than the sigmoid activation. 

```python 
number_hidden_layers = 1
```

```python 
model = Sequential()
model.add(Dense(512, input_shape=(784,), activation='relu')) 
model.add(Dropout(0.2))

while number_hidden_layers > 1:
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    number_hidden_layers -= 1


model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(),
              metrics=['accuracy'])

# Training (You don't need to change this part of the code)
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=0, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```
![model_hl1](model_hl_1.PNG)
![plot_hl1](graph_epoch_hl1.PNG)


[Back to Table of Contents](#toc)

<a name="hidden_layer"/> 

## Add hidden layers to check model accuracy

By increasing the hidden layer only with activation function ReLU the accuracy improves, but we see that it is not the same for sigmoid activation function(it decreases significantly) 

```python
# Use this parameter to change the depth of the model
number_hidden_layers = 3  # Number of hidden layers

# Model
# Remaining parameter are the same for building the model 
# Training (You don't need to change this part of the code)
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=0, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# list all data in history
# summarize history for accuracy
```

![model_hl1](model_hl_3.PNG)
![plot_hl1](graph_epoch_hl3.PNG)

[Back to Table of Contents](#toc)

<a name="cal_code"/> 

## Challenge - MNIST Caluculator Code 

Time for the challenge to build the Calculator. 
1. First we take a list of random images from our dataset  
1. Determine the value of the MNIST image using CNN model 
1. Use numpy libraries to append the values 

Following is the python code 
```python 
# To generate addition 
def gen_test():
    
    indices = np.random.randint(0, y_train.shape[0], size=10)
        
    q = np.array([x_train[i] for i in indices]).reshape(10, 28, 28, 1)
    n = np.array([ np.argmax(y_train[i]) for i in indices])
    s = n.cumsum()
    a = sum(n)
    
    return (q, n, s, a)

# Define digits 
def generate_custom_test(digits):
    
    q = []
    
    for digit in digits:
        s = -1

        while digit != s:
            index = np.random.randint(0, y_train.shape[0], size=1)[0]
            s = np.argmax(y_train[index])
                
        q.append(x_train[index])
         
    q = np.array(q).reshape(10, 28, 28, 1)
    
    return (q, digits, np.cumsum(digits), sum(digits))

# Inspect model performance 
def inspect_test(model, q, n, s, a):
    
    prediction = model.predict(np.array([ q ]))

    for i, digit in enumerate(q):
        plt.subplot(1, 10, i + 1)
        plt.axis('off')
        plt.imshow(digit.reshape(28,28), cmap='gray', interpolation='none')
    
    print('Question: {}'.format(n))
    plt.show()
    
    print('Real answer:', a)
    print('Predicted answer:', prediction.flatten()[9]);
```

[Back to Table of Contents](#toc)

<a name="cal_sol"/> 

## Challenge - MNIST Calculator Solution

Following is the solution to our MNIST calculator. The aim is to design and train a model that produces a sum of a sequence of MNIST images. Let us consider a sequence of MNIST images ranging from 0 to 9 for example, the real cumulative sum of the sequence is 45 and you can see our model predicts 45.5 (slightly overfitting) which is extremely close.

![cal_sol](cal_sol.PNG)

[Back to Table of Contents](#toc)

<a name="concl"/> 

## Conclusion 

1. sequential: It is to initiate the model
1. Convolution Layer: It is used in recognizing the digits. We define 32 filters for the first 2 convolutional layer and later define 64 filters, we use 33 feature detector to scan the original image. The input specifies 10,28,28,1 where 10 is denotes by the number of images, 2828 is the image size we define and 1 is the number of channel, here we have grey scale image hence only 1 channel.
1. Batch Normalization: It maintains the mean activation as close to zero and standard deviation as close to 1.
Activation Layer: We use "relu" as the activation function as it has been proved to give good accuracy as per our above analysis in 2nd part of the question.
1. Max pooling: It helps in selecting the most relevant features from our image by dealing with spatial invariance. i.e. slanting in the hand written digits.
1. Flatten: To end up with a 1D array.
1. Dropout: Helps in regularization and avoid overfitting, we set activation of a neuron with probability less than 0.3 to zero. i.e. we omit those neurons.
1. Optimizer - Adam: It combines the advantage of two other extensions of stochastic gradient descent i.e. Adaptive gradient and Root mean square propogation making it more efficient.

We see that the model slightly over fits the data, but our solution is very close to the actual cumulative sum for the numbers. 
