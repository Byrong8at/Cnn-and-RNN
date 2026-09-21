import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class MaxPool2:
    def iterate_regions(self, image):
        '''
        Generates all non-overlapping 2x2 patches of the input image for max pooling.
        - image: 3D NumPy array with shape (height, width, num_filters)
        Returns:
        - patches: 4D NumPy array of shape (height/2, width/2, 2, 2, num_filters)
        containing the 2x2 patches for each channel
        '''
        patches = sliding_window_view(image, (2, 2), axis=(0,1))  # shape: (h-1, w-1, 2, 2, num_filters)
        patches = patches[::2, ::2, :, :, :]  # take every second patch to ensure non-overlapping
        patches = patches.transpose(0, 1, 3, 4, 2)#celle la
        return patches

    def forward(self, input):
        '''
        Performs a forward pass of the max pooling layer.

        Input:
        - input: 3D NumPy array with shape (height, width, num_filters)

        Returns:
        - output: 3D NumPy array with shape (height/2, width/2, num_filters),
                containing the maximum value from each non-overlapping 2x2 patch
                for every channel.

        Logic:
        - Extract all non-overlapping 2x2 patches using iterate_regions.
        - Compute the maximum value in each patch across the height and width dimensions.
        - Store the result in the output array, preserving the channel dimension.
        '''
        self.last_input = input  # store input for backpropagation

        patches = self.iterate_regions(input)  # shape: (h/2, w/2, 2, 2, num_filters)
        output = np.amax(patches, axis=(2, 3))  # max over the 2x2 regions, shape: (h/2, w/2, num_filters)

        return output

    def backprop(self, d_out):
        '''
        Performs the backward pass of the max pooling layer.

        Input:
        - d_out: gradient of the loss with respect to the output of this layer,
                with shape (h/2, w/2, num_filters)

        Returns:
        - d_input: gradient of the loss with respect to the input of this layer,
                with the same shape as the original input (h, w, num_filters)

        Logic:
        1. For each 2x2 patch in the input (from forward pass), determine which
        element(s) were the maximum.
        2. Assign the corresponding gradient from d_out only to these max positions.
        3. Accumulate contributions from all patches into the d_input array.
        '''
        patches = self.iterate_regions(self.last_input)          # (h/2, w/2, 2, 2, C)
        amax = np.max(patches, axis=(2, 3))                      # (h/2, w/2, C)

        mask = (patches == amax[:, :, None, None, :])            # (h/2, w/2, 2, 2, C)
        d_patches = mask * d_out[:, :, None, None, :]            # (h/2, w/2, 2, 2, C)

        h_out, w_out, _, _, C = d_patches.shape

        # Base grid of patch positions
        I = np.repeat(np.arange(h_out), w_out)   # (h_out*w_out)
        J = np.tile(np.arange(w_out), h_out)     # (h_out*w_out)

        # Convert patch coordinates → input coordinates
        row0 = 2 * I
        col0 = 2 * J

        # Offsets inside 2×2
        roff = np.array([0, 0, 1, 1])
        coff = np.array([0, 1, 0, 1])

        rows = (row0[:, None] + roff).reshape(-1)
        cols = (col0[:, None] + coff).reshape(-1)

        # Flatten to match indices
        flat = d_patches.reshape(-1, C)           # ((h_out*w_out*4), C)

        # Allocate gradient
        d_input = np.zeros_like(self.last_input)

        # Add in one shot for all channels
        for c in range(C):
            np.add.at(d_input, (rows, cols, c), flat[:,c])

        return d_input