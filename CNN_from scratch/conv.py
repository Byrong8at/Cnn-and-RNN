import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class Conv3x3x3:
    def __init__(self, num_filters,inputs=3):
        '''
        Initialize convolutional layer with 3x3 filters.
        
        Each filter is a 3D numpy array with dimensions (3, 3, 3) for RGB channels.
        - num_filters: number of filters in the layer.
        
        We use a random generator with a fixed seed for reproducibility, 
        so the initial filters are the same every time.
        '''
        self.num_filters = num_filters  # store number of filters
        self.inputs=inputs
        
        rng = np.random.default_rng(42)  # create a Generator with seed 42
        fan_in = 3 * 3 * 3
        scale = np.sqrt(2.0 / fan_in)
        self.filters = rng.standard_normal((num_filters, 3, 3, inputs)) * scale
        #self.filters = rng.standard_normal((num_filters, 3, 3, 3))  # shape: (num_filters, 3,3,3)
        
    
    def iterate_regions(self, image):
        '''
        Generate all possible 3x3 regions (patches) of the input image.

        Returns:
        - patches: numpy array of shape (h-2, w-2, 3, 3, C) where each (3,3,C) slice is a patch.
        '''
        patches = sliding_window_view(image, (3,3), axis=(0,1)) # shape: (h-2, w-2, 3, 3, C)
        return patches


    def forward(self, input):
        '''
        Forward pass of the convolutional layer.

        Input:
        - input: 3D array of shape (h, w, c)

        Returns:
        - output: 3D array of shape (h-2, w-2, num_filters)

        Logic:
        - Extract all 3*3*C patches at once using sliding windows.
        - Flatten each patch and each filter to allow vectorized dot products.
        - Use a single batched operation to compute the response of every filter
          on every patch, producing the full feature maps.
        '''
        self.last_input = input  # store input for backprop

        patches = self.iterate_regions(input)  # shape: (h-2, w-2, 3, 3, c)
        patches = patches.reshape(patches.shape[0], patches.shape[1], -1)  # shape: (h-2, w-2, 9*c) => (h-2, w-2, 27)
        
        F = self.filters.shape[0]  # number of filters
        filters_reshaped = self.filters.reshape(F, -1)  # shape: (F, 27)

        output = np.einsum("ijc, fc -> ijf", patches, filters_reshaped) # shape: (h-2, w-2, F)

        return output


    def backprop(self, d_out, learn_rate):
        '''
        Backward pass of the convolutional layer.

        Input:
        - d_out: gradient of the loss with respect to the layer's output,
                shape (h-2, w-2, num_filters)
        - learn_rate: learning rate for updating the filters

        Returns:
        - d_input: gradient of the loss with respect to the layer's input,
                shape (h, w, c)

        Logic:
        1. Compute gradients for each filter by multiplying each input patch with
        the corresponding output gradient and summing over all positions.
        2. Update the filters using gradient descent.
        3. Compute the input gradients by propagating each filter's contribution 
        to all positions it covers in the input, accumulating overlapping contributions.
        '''
        # Compute filter gradients
        b_patches = self.iterate_regions(self.last_input)
        b_patches = b_patches.reshape(b_patches.shape[0], b_patches.shape[1], -1) # shape: (h-2, w-2, 27)

        # Compute gradients for each filter using einsum for efficiency
        d_filters = np.einsum("ijp, ijf -> fp", b_patches, d_out) # shape: (num_filters, 27)

        # Gradient descent update
        self.filters -= learn_rate * d_filters.reshape(self.filters.shape) # reshape d_filters back to (num_filters, 3, 3, 3)

        d_input = np.zeros_like(self.last_input)  # initialize gradient w.r.t. input

        for i in range(d_out.shape[0]):  # iterate over output height
            for j in range(d_out.shape[1]):  # iterate over output width
                for f in range(self.num_filters):  # iterate over filters
                    # The gradient contribution from filter f at position (i,j)
                    gradient = d_out[i, j, f] * self.filters[f]  # shape: (3, 3, 3)
                    d_input[i:i+3, j:j+3] += gradient  # accumulate into the correct slice of d_input

        return d_input