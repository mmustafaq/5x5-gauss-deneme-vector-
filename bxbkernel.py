# -*- coding: utf-8 -*-
"""
Created on Thu May 25 22:16:44 2023

@author: mmust
"""

import numpy as np
import cv2


import numpy as np
n = 256
def apply_gaussian_filter(input_array):
    output_array = np.zeros_like(input_array, dtype=np.int32)  # Create an array to store filtered values
    
    # Define the Gaussian kernel array
    gaussian_kernel = np.array([1,4,7,4,1,4,16,26,16,4,7,26,41,26,7,4,16,26,16,4,1,4,7,4,1])
    
    # Iterate over each element in the input array
    for i in range(254*254):
        #print(i)
        # Calculate the weighted sum of neighboring elements using the Gaussian kernel
        output = (input_array[0 + i] * gaussian_kernel[0]+input_array[1 + i] * gaussian_kernel[1]+input_array[2 + i] * gaussian_kernel[2]+input_array[3 + i] * gaussian_kernel[3]+input_array[4 + i] * gaussian_kernel[4]
        +input_array[n + i] * gaussian_kernel[5]+input_array[n+1 + i] * gaussian_kernel[6]+input_array[n+2 + i] * gaussian_kernel[7]+input_array[n+3 + i] * gaussian_kernel[8]+input_array[n+4 + i] * gaussian_kernel[9]
        +input_array[2*n + i] * gaussian_kernel[10]+input_array[2*n+1 + i] * gaussian_kernel[11]+input_array[2*n+2 + i] * gaussian_kernel[12]+input_array[2*n+3 + i] * gaussian_kernel[13]+input_array[2*n +4 + i] * gaussian_kernel[14]
        +input_array[3*n + i] * gaussian_kernel[15]+input_array[3*n+1 + i] * gaussian_kernel[16]+input_array[3*n+2 + i] * gaussian_kernel[17]+input_array[3*n+3 + i] * gaussian_kernel[18]+input_array[3*n +4 + i] * gaussian_kernel[19]
        +input_array[4*n + i] * gaussian_kernel[20]+input_array[4*n+1 + i] * gaussian_kernel[21]+input_array[4*n+2 + i] * gaussian_kernel[22]+input_array[4*n+3 + i] * gaussian_kernel[23]+input_array[4*n +4 + i] * gaussian_kernel[24])/273
        
        output_array[n+2+ i] = output
    return output_array

# Assuming you have a 4096 size integer array called 'input_array'

filtered_array = apply_gaussian_filter(input_array)
reshaped_array = filtered_array.reshape((n,n))
reshaped_array = reshaped_array.astype(np.uint8)

input_array = np.array(input_array)
reshaped_array1 = input_array.reshape((n, n))
reshaped_array1 = reshaped_array1.astype(np.uint8)
# Display the image
cv2.imshow('Image', reshaped_array)
cv2.imshow('Imaged', reshaped_array1)
cv2.waitKey(0)
cv2.destroyAllWindows()