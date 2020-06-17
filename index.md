## Image classification using Convolutional Neural Network 

This project explain basics of backpropogation, concept of hidden layers and building a Convolutional neural network model for MNIST dataset. MNIST dataset is a collection of images of handwritten digits. The project tries to classify each image into its appropriate number from 0 to 9. 

But there is a little challenge in the end.. What is the challenge ? 
We also check the accuracy of the model by adding random 10 digits from the dataset to obtain the sum of those 10 digits. 
For example: adding numbers 1+2+3+4+5+6+7+8+9+0 should give us 45 right? So, We pick random digits from the MNIST datasets which have been classified as 0,1,2 till 9 and then take the sum of those digits to check if our generated model also provides the sum as 45. 
Therefore, testing the accuracy of the model, if the sum is more than 45 then our model overfitts the data, otherwise it underfits the data. 

## What is backpropogation in neural network ? 
For a classification problem in supervised learning, the following illustrates a hand made graph for backpropogation. ![backpro](backpropo.PNG)
