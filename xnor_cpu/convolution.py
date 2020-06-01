from scipy import ndimage
import numpy as np 


# %%
# Generate a random array as image
image = np.random.rand(256,256)
image = np.round(image)
# Generate a random array as convolution kernel
kernel = np.random.rand(3,3)
kernel = np.round(kernel)
# save generated array
image = image.astype(np.uint8)
path_width = (np.array(kernel.shape) - 1) / 2
path_width_1 = (path_width[0], path_width[0])
path_width_2 = (path_width[1], path_width[1])
zero_pad = np.pad(image, pad_width=((int(path_width[0])), (int(path_width[0]))), mode='constant', constant_values = 0)
kernel = kernel.astype(np.uint8)
# %%
np.savetxt('Input_image.txt',zero_pad, header=f'{zero_pad.shape}', fmt='%i', delimiter=',')
np.savetxt("Convolution_kernel.txt", kernel, header=f'{kernel.shape}', fmt='%i', delimiter=',')
# %%
# apply convolution
result = ndimage.filters.convolve(image, kernel, mode='constant', cval=0)
np.savetxt("Result_image.txt", result, header=f'{result.shape}', fmt='%i', delimiter=',')
# read result of cpp code and find error if exists