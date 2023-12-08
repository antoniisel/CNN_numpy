import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, softmax, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """

    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers

        self.conv1 = ConvolutionalLayer(in_channels=input_shape[-1], 
                                        out_channels=conv1_channels, 
                                        filter_size=3, padding=1)
        self.ReLU = ReLULayer()
        self.conv2 = ConvolutionalLayer(in_channels=conv1_channels, 
                                        out_channels=conv2_channels, 
                                        filter_size=3, padding=1)
        
        self.MaxPool = MaxPoolingLayer(4, 4)
        self.Flatten = Flattener()
        self.FC = FullyConnectedLayer(n_input=((np.prod(input_shape[:2])) // 16) * conv2_channels,
                                       n_output=n_output_classes)

        

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # we clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        #  I do not imlplement L2 regularization
        #  in this project

        for param in self.params().values():
            param.grad = np.where(False, param.grad, 0)

        layers = (self.conv1, self.conv2, self.MaxPool, self.Flatten, self.FC)
        X = X.copy()
        y = y.copy().reshape(-1,1)

        for layer in layers:
            
            X = layer.forward(X)

        loss, d_input = softmax_with_cross_entropy(X, y)

        for layer in layers[::-1]:
            
            d_input = layer.backward(d_input)
        


        return loss

    def predict(self, X):
        layers = (self.conv1, self.conv2, self.MaxPool, self.Flatten, self.FC)
        X = X.copy()
        for layer in layers:
            
            X = layer.forward(X)

        probs = softmax(X)
        
        pred = np.argmax(probs, axis=1)
        
        return pred

    def params(self):
        result = {'C1.W': self.conv1.params()['W'], 
                  'C1.B': self.conv1.params()['B'],
                  'C2.W': self.conv2.params()['W'], 
                  'C2.B': self.conv2.params()['B'], 
                  'FC.W': self.FC.params()['W'], 
                  'FC.B': self.FC.params()['B']}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        

        return result
