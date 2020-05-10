import numpy as np
import random
import math


def convParam_gen(batch_size, in_dim, kernel_size, in_channels, out_channels, stride, out_dim, bias, depthwidth, pool_size, pool_stride, out_dim_pooled, output_scale):
    
    # initialize all parameters
    padding = math.floor((out_dim*stride + kernel_size - 1 - in_dim)/2)
    n_patches =  math.ceil(in_dim/stride)* math.ceil(in_dim/stride) * batch_size
    patch_size = kernel_size * kernel_size * in_channels
    pool_padding = math.floor((out_dim_pooled*pool_stride + pool_size - 1 - out_dim)/2)
    I = n_patches
    J = out_channels
    K = patch_size

    # generating weight and bias
    weight_arr = np.random.randint(20, size = (patch_size, out_channels))

    # generating bias
    if(bias):
        bias_arr = np.random.randint(5, size = out_channels)
    else:
        bias_arr = np.zeros((out_channels,), dtype = int)

    return padding, n_patches, patch_size, pool_padding, I, J, K, weight_arr, bias_arr

def image_gen(batch_size, in_dim, in_channels):
    imag_arr = np.random.randint(20, size = (batch_size, in_dim, in_dim, in_channels))

    return imag_arr




## generating parameter files
f = open("neuralODE_1layer_params.h", "w+")

f.write("#ifndef NEURALODE_PARAMETERS_H\n")
f.write("#define NEURALODE_PARAMETERS_H\n\n")
f.write("#include <include/gemmini_params.h>\n#include <stdbool.h>\n\n")

## layer 1
conv1_batch_size = 64
conv1_in_dim = 3
conv1_kernel_size = 3
conv1_in_channels = 64
conv1_out_channels = 64
conv1_stride = 1
conv1_bias = 1
conv1_depthwidth = 0
conv1_out_dim = 3
conv1_pool_size = 1
conv1_pool_stride = 1
conv1_out_dim_pooled = 3
conv1_output_scale = 1
conv1_padding, conv1_n_patches, conv1_patch_size, conv1_pool_padding, conv1_I, conv1_J, conv1_K, conv1_weight_arr, conv1_bias_arr = convParam_gen(conv1_batch_size, conv1_in_dim, conv1_kernel_size, conv1_in_channels, conv1_out_channels, conv1_stride, conv1_out_dim, conv1_bias, conv1_depthwidth, conv1_pool_size, conv1_pool_stride, conv1_out_dim_pooled, conv1_output_scale)

f.write("static const elem_t conv_1_w[" + str(conv1_patch_size) + "][" + str(conv1_out_channels) + "] row_align(1) = {")
for i in range(conv1_patch_size):
    for k in range(conv1_out_channels):
        if(k == 0):
            f.write("{" + str(conv1_weight_arr[i,k]) + ",")
        elif(k == conv1_out_channels - 1):
            f.write(str(conv1_weight_arr[i,k]) + "}")
            if(i < conv1_patch_size - 1):
                f.write(",")
        else:
            f.write(str(conv1_weight_arr[i,k]) + ",") 
f.write("};\n")

f.write("static const acc_t conv_1_b["+ str(conv1_out_channels) + "] row_align(1) = {")
for i in range(conv1_out_channels):
    if(i < conv1_out_channels-1):
        f.write(str(conv1_bias_arr[i]) + ",")
    else:
        f.write(str(conv1_bias_arr[i]))
f.write("};\n")

f.write("static elem_t conv_1_in[" + str(conv1_n_patches) +"][" + str(conv1_patch_size) +"] row_align(1);\n")
f.write("static elem_t conv_1_out[" + str(conv1_n_patches) +"][" + str(conv1_out_channels) +"] row_align(1);\n")
f.write("static const struct ConvParams conv_1_params = {.batch_size=" + str(conv1_batch_size) + ", .in_dim=" + str(conv1_in_dim) + ", .kernel_size=" + str(conv1_kernel_size) + ", .in_channels=" + str(conv1_in_channels) + ", .out_channels=" + str(conv1_out_channels))
f.write(", .stride=" + str(conv1_stride) + ", .padding=" + str(conv1_padding) + ", .bias=" + str(conv1_bias) + ", .depthwise=" + str(conv1_depthwidth) + ", .out_dim=" + str(conv1_out_dim) + ", .n_patches=" + str(conv1_n_patches) + ", .patch_size=" + str(conv1_patch_size))
f.write(", .pool_size=" + str(conv1_pool_size) + ", .pool_stride=" + str(conv1_pool_stride) + ", .pool_padding=" + str(conv1_pool_padding) + ", .out_dim_pooled=" + str(conv1_out_dim_pooled) + ", .output_scale=" + str(conv1_output_scale) + ", .I=" + str(conv1_I))
f.write(", .J=" + str(conv1_J) + ", .K=" + str(conv1_K) + ", .res_scale=0};\n\n")

f.write("#endif")

f.close()


## generate input activation files
f = open("neuralODE_1layer_images.h", "w+")

in_imag = image_gen(conv1_batch_size, conv1_in_dim, conv1_in_channels)

f.write("#ifndef NEURALODE_IMAGES_H\n")
f.write("#define NEURALODE_IMAGES_H\n\n")
f.write("static const elem_t images[" + str(conv1_batch_size) + "][" + str(conv1_in_dim) + "][" + str(conv1_in_dim) + "][" + str(conv1_in_channels) + "] row_align(1) = ")
for i in range(conv1_batch_size):
    if(i == 0):
        f.write("{")
    for k in range(conv1_in_dim):
        if(k == 0):
            f.write("{")
        for m in range(conv1_in_dim):
            if(m == 0):
                f.write("{")
            for n in range(conv1_in_channels):
                if(n == 0):
                    f.write("{")
                if(n < conv1_in_channels-1):
                    f.write(str(in_imag[i,k,m,n]) + ",")
                else:
                    f.write(str(in_imag[i,k,m,n]))
            if(m < conv1_in_dim-1):
                f.write("},")
            else:
                f.write("}")
        if(k < conv1_in_dim-1):
            f.write("},\n")
        else:
            f.write("}")
    if(i < conv1_batch_size-1):
        f.write("},\n")
    else:
        f.write("}")
f.write("};\n\n")
f.write("#endif")

f.close()

## initialize conv params


