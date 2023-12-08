import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    loss = reg_strength * np.sum(W**2)
    grad = reg_strength * 2 * W

    return loss, grad
    


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # final implementation shouldn't have any loops

    predictions = predictions.copy()
    predictions -= np.max(predictions, axis=-1).reshape(-1, 1)
    total = np.sum(np.exp(predictions), axis=1).reshape(-1, 1)
    probs = np.exp(predictions) / total

    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # final implementation shouldn't have any loops
    
    target_index = target_index
        
    q = np.take_along_axis(probs, target_index, axis=1)
        

    loss = - np.sum(np.log(q), axis=1)

    return np.mean(loss)


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    
    # TODO implement softmax with cross-entropy
    # final implementation shouldn't have any loops
  
    probs = softmax(predictions)
    
         
    loss = cross_entropy_loss(probs, target_index)

    probs[np.arange(probs.shape[0])[:, None], target_index] -= 1

    dprediction = probs

    
    return loss, dprediction / dprediction.shape[0] 

class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        
        self.zeros = (X >= 0)
        Y = np.where(X >= 0, X, 0)
        
        
        return Y

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """


        # TODO: Implement backward pass
        # final implementation shouldn't have any loops
        d_result = d_out * self.zeros

        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None
     

    def forward(self, X):
        # TODO: Implement forward pass
        # final implementation shouldn't have any loops

        self.X = X.copy()
        return self.X @ self.W.value + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute


        d_input = d_out @ self.W.value.T
        self.B.grad += np.sum(d_out, axis=0)
        self.W.grad += self.X.T @ d_out

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding
        self.X = None


    def forward(self, X):
        X = X.copy()
        X = pad_array(X, pad_height=self.padding, pad_width=self.padding)

        batch_size, height, width, channels = X.shape

        out_height = height  - (self.filter_size - 1)
        out_width = width - (self.filter_size - 1)
        self.X = X
        
        # TODO: Implement forward pass
        # We should setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # We use loops for going over width and height
        # but try to avoid having any other loops

        W_mat = self.W.value.reshape(-1, self.out_channels)
        
        
        Y = np.zeros((batch_size, out_height, out_width, self.out_channels))
        
        for y in range(out_height):
            for x in range(out_width):
                
                border_x = x + self.filter_size
                border_y = y + self.filter_size
              
                Y[:, x, y, :] = X[:, x:border_x, y:border_y, :].reshape(batch_size, -1) @ W_mat + self.B.value

        
        return Y


    def backward(self, d_out):
        # Forward pass was reduced to matrix multiply

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # We try to avoid having any other loops here too

   

        W_mat = self.W.value.reshape(-1, self.W.value.shape[-1])
        d_input = np.zeros(self.X.shape)

        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
              
                
                border_x = x + self.filter_size
                border_y = y + self.filter_size

                d_input_xy = d_input[:, x:border_x, y:border_y, :].reshape(batch_size, -1)
                d_out_xy = d_out[:, x, y, :]
                X_xy = self.X[:, x:border_x, y:border_y, :].reshape(batch_size, -1)


                d_input_xy += d_out_xy @ W_mat.T
                d_input[:, x:border_x, y:border_y, :] =  d_input_xy.reshape(batch_size, self.filter_size, 
                                                                                self.filter_size, channels)
               
                self.W.grad += (X_xy.T @ d_out_xy).reshape(self.W.value.shape)
                self.B.grad += np.sum(d_out_xy, axis=0).reshape(self.B.value.shape)
             

        
        return un_pad_array(d_input, pad_height=self.padding, pad_width=self.padding)
    
    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.indeces = {}
        

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Similarly to Conv layer, loop on
        # output x/y dimension

        self.X = X.copy()

        out_height = ((height  - (self.pool_size - 1))) // self.stride + bool(((height  - (self.pool_size - 1))) % self.stride)
        out_width =  ((width  - (self.pool_size - 1))) // self.stride + bool(((width  - (self.pool_size - 1))) % self.stride)

        Y = np.zeros((batch_size, out_height, out_width, channels))

        
        for y in range(out_height):
            for x in range(out_width):
                border_y = y * self.stride + self.pool_size
                border_x = x * self.stride + self.pool_size

                A = X[:, x * self.stride : border_x, y * self.stride :border_y, :]
                Y[:, x, y, :] = np.max((A.reshape(A.shape[0], A.shape[1]*A.shape[2], -1)), axis=1)
                               

        return Y

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        d_in = np.zeros(self.X.shape)

        for y in range(out_height):
            for x in range(out_width):
                
                border_y = y * self.stride + self.pool_size
                border_x = x * self.stride + self.pool_size

                A = self.X[:, x * self.stride : border_x, y * self.stride :border_y, :]
                d_out_xy = d_out[:, x, y, :]

                shape_0 = np.arange(A.shape[0]).reshape(-1, 1)
                shape_1 = np.argmax(A.reshape(A.shape[0], A.shape[1]*A.shape[2], -1), axis=1)
                shape_2 = np.array(list(range(A.shape[-1])) * A.shape[0]).reshape(A.shape[0], A.shape[-1])

                d_in_xy = d_in[:, x * self.stride : border_x, y * self.stride :border_y, :].reshape(A.shape[0], A.shape[1]*A.shape[2], -1)

                

                d_in_xy[shape_0, shape_1, shape_2] += d_out_xy

                d_in[:, x * self.stride : border_x, y * self.stride :border_y, :] = d_in_xy.reshape(A.shape)


        return d_in

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X_shape = X.shape
        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        
        return X.reshape((batch_size, height*width*channels))

    def backward(self, d_out):
        # TODO: Implement backward pass

        d_in = d_out.reshape(self.X_shape)

        return d_in

    def params(self):
        # No params!
        return {}


def pad_array(array, pad_height, pad_width):
    # Get the dimensions of the array
    batch_size, height, width, input_channels = array.shape
    
    # Calculate the new dimensions after padding
    new_height = height + 2 * pad_height
    new_width = width + 2 * pad_width
    
    # Create a new array with the padded dimensions
    padded_array = np.zeros((batch_size, new_height, new_width, input_channels))
    
    # Compute the indices to slice the original array into the padded array
    h_start = pad_height
    h_end = pad_height + height
    w_start = pad_width
    w_end = pad_width + width
    
    # Copy the original array into the padded array
    padded_array[:, h_start:h_end, w_start:w_end, :] = array
    
    return padded_array

def un_pad_array(array, pad_height, pad_width):
    # Get the dimensions of the array
    batch_size, height, width, input_channels = array.shape
    
    # Calculate the new dimensions after padding
    new_height = height - 2 * pad_height
    new_width = width - 2 * pad_width
    
    
    # Compute the indices to slice the original array into the padded array
    h_start = pad_height
    h_end = pad_height + new_height
    w_start = pad_width
    w_end = pad_width + new_width
    
    # Copy the original array into the padded array
    un_padded_array = array[:, h_start:h_end, w_start:w_end, :]
    
    return un_padded_array