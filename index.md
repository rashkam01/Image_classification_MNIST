## Image classification using Convolutional Neural Network 

This project explain basics of backpropogation, concept of hidden layers and building a Convolutional neural network model for MNIST dataset. MNIST dataset is a collection of images of handwritten digits. The project tries to classify each image into its appropriate number from 0 to 9. 

But there is a little challenge in the end.. What is the challenge ? 
We also check the accuracy of the model by adding random 10 digits from the dataset to obtain the sum of those 10 digits. 
For example: adding numbers 1+2+3+4+5+6+7+8+9+0 should give us 45 right? So, We pick random digits from the MNIST datasets which have been classified as 0,1,2 till 9 and then take the sum of those digits to check if our generated model also provides the sum as 45. 
Therefore, testing the accuracy of the model, if the sum is more than 45 then our model overfitts the data, otherwise it underfits the data. 

## What is backpropogation in neural network ? 
For a classification problem in supervised learning, the following illustrates a hand made graph for backpropogation. ![backpro](backpropo.PNG)
Considering it has single input e.g. x, (x is a pixel value that get fed into the neural network). The output neurons are typically the number of classes into which we classify the image. 
Loss funtion is the prediction of error of Nerual Net. Activation function defines the output of the node given a certain input to the node. Learning rate defines the step size, the rate at which the function moves to the local minima. 
Now, given that the learning rate is 0.1, activation function ReLU, a loss function of MSE (mean square error) we see the computation behind a backpropogation. 

1. Compute graph: provided above
2. Choice of Loss Function is MSE(as per above graph) L = $\frac{1}{2} (o - y)^2$ = $\frac{1}{2} (24 - 3)^2$ = 220.5
3. Partial derivative and updated expression for each of the parameters for each of the data-points: 
$$\frac{\partial L}{\partial o} =  (o - y) =  24 - 3 = 21 $$
$$\frac{\partial L}{\partial c} = \frac{\partial o}{\partial c}* \frac{\partial L}{\partial o} = 4 * 21 = 84 $$
$$ \frac{\partial L}{\partial e} = \frac{\partial o}{\partial e}* \frac{\partial L}{\partial o} = 1 * 21 = 21  $$
$$ \frac{\partial L}{\partial a} = \frac{\partial c}{\partial a}* \frac{\partial L}{\partial c} = 1 * 84 = 84  $$
$$ \frac{\partial L}{\partial b} = \frac{\partial e}{\partial b}* \frac{\partial L}{\partial e} = 2 * 21 = 42  $$
$$ \frac{\partial L}{\partial w_5} = \frac{\partial o}{\partial w_5}* \frac{\partial L}{\partial o} = 8 * 21 = 168  $$
$$ \frac{\partial L}{\partial w_4} = \frac{\partial o}{\partial w_4}* \frac{\partial L}{\partial o} = 4 * 21 = 84  $$
$$ \frac{\partial L}{\partial w_3} = \frac{\partial e}{\partial w_3}* \frac{\partial L}{\partial e} = 4 * 21 = 84  $$
$$ \frac{\partial L}{\partial w_2} = \frac{\partial c}{\partial w_2}* \frac{\partial L}{\partial c} = 4 * 84 = 336  $$
$$ \frac{\partial L}{\partial w_1} = \frac{\partial a}{\partial w_1}* \frac{\partial L}{\partial a} + \frac{\partial b}{\partial w_1}* \frac{\partial L}{\partial b} = 2*84 + 2*42 = 252  $$

4. By formula we know, 
$$w \leftarrow w - \alpha * \frac{\partial}{\partial w} L(x, y; w, b)$$
and learning rate is given as 0.1 and there is no bias therefore, 

$$ w_1 \leftarrow w_1 - 0.1 * 252 = 2 - 0.1 * 252 = -23.2 $$ 
$$  w_2 \leftarrow w_2 - 0.1 * 336 = 1 - 0.1 * 336 = -32.6 $$ 
$$  w_3 \leftarrow w_3 - 0.1 * 84 = 2 - 0.1 * 84 = -6.4 $$ 
$$ w_4 \leftarrow w_4 - 0.1 * 84 = 4 - 0.1 * 84 = -4.4 $$ 
$$ w_5 \leftarrow w_5 - 0.1 * 168 = 1 - 0.1 * 168 = -15.8 $$ 
