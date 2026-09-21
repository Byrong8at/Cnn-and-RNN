import numpy as np
from model import *

class Flatten():
    def __init__(self):
        """
        Initialize the input shape to None.
        """
        self.input_shape = None

    def forward(self,x):
        """
        flatten the input into 1D vector
        """
        self.input_shape=x.shape
        return x.flatten()
    
    def backprop(self,d):
        """
        transform the flatten layer back into 3D
        """
        return d.reshape(self.input_shape)

class Dense():
    def __init__(self,input_size,output_size):
        """
        Initialize the different value
        """
        scale = np.sqrt(2.0 / input_size)#add scale
        self.weight = np.random.randn(input_size, output_size) * scale
        self.bias=np.zeros(output_size)
    def forward(self,input):
        self.Y=np.dot(input, self.weight)+self.bias
        self.last_input = input  # store input for backprop
        return self.Y
    def backprop(self, d_out, learn_rate,l2_lambda_):
        self.last_input=self.last_input.reshape(-1, 1)
        d_out=d_out.reshape(1, -1)
        d_weight=np.dot(self.last_input,d_out)
        d_input=np.dot(d_out,self.weight.T)
        self.weight -= learn_rate * (d_weight + l2_lambda_ * self.weight)
        self.bias -= learn_rate * d_out.flatten()
        return d_input
    
class Dropout():
    def __init__(self, drop_rate):
        """
        Drop rate need to be between 0 and 1
        """
        self.rate = 1 - drop_rate
        self.binary_mask = None

    def forward(self, inputs, is_training=True):
        """
        inputs: input data
        is_training: If True, apply dropout. If False (test), pass data through.
        """
        if is_training:
            # Create the binary mask with binomial function
            # 1 is to force the result is only 0 or 1
            # rate is Parameter of the distribution between 0 and 1
            # We divide by self.rate to maintain the same energy (Inverted Dropout)
            self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
            return inputs * self.binary_mask
        else:
            # During validation/test, we do NOT drop any neurons
            return inputs
    
    def backprop(self, d_out):
        # Apply the mask to the gradient
        return d_out * self.binary_mask