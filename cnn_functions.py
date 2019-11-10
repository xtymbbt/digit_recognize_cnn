import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import time
import math

# load data function:
def load_data(filename):
    print("Loading data...")
    start = time.time()
    csv = np.array(pd.read_csv(filename))
    y = np.array(csv[:, 0], ndmin=2).T
    x = np.array(csv[:, 1:])
    end = time.time()
    print("Done.")
    print("Execute time is: ", (end-start), "seconds")
    return x, y


# padding layer:
def padding(input_matrix, padding_size):
    # input_matrix must be of shape: (sample_number, row_pixel, column_pixel, channel_number)
    # padding_size must be a scalar
    # padding_matrix must be a matrix: shape: (dimension_number, 2)
    # print("Zeros padding...")
    # start = time.time()
    padding_matrix = np.ones((4, 2)) * padding_size
    padding_matrix[0, :] = padding_matrix[0, :] * 0
    padding_matrix[3, :] = padding_matrix[3, :] * 0
    padding_matrix = tf.cast(padding_matrix, tf.int64)
    # m = input_matrix.shape[0]
    # row_pixel = input_matrix.shape[1]
    # column_pixel = input_matrix.shape[2]
    # channel = input_matrix.shape[3]
    # x_temp = np.zeros((m, row_pixel+2, column_pixel+2, channel))
    # for i in range(m):
    #     for j in range(channel):
    #         x_temp[i, :, :, j] = tf.pad(input_matrix[i, :, :, j], padding_matrix, mode='CONSTANT')
    # output_matrix = x_temp
    output_matrix = tf.pad(input_matrix, padding_matrix, mode='CONSTANT')
    # end = time.time()
    # print("Done.")
    # print("Execute time is: ", (end-start), "seconds")
    return output_matrix


# convolution layer using tensor flow built-in function:
def convolution_layer_tf(input_matrix, convolution_filter, strides):
    # input_matrix must be of shape: (sample_number, row, column, channel_number)
    # since input_matrix has been padded, it's row and column will be larger than before.
    # row = row + padding size * 2
    # column = row + padding size * 2
    # convolution_filter must be of shape: (n, n, channel_number, convolution_filter_channel_number)
    # strides must be a list: (sample_number_stride, row_stride, column_stride, channel_number_stride)
    # print("Calculating Convolution...")
    # start = time.time()
    y = tf.nn.conv2d(input_matrix, convolution_filter, strides=strides, padding='SAME')
    # end = time.time()
    # print("Done.")
    # print("Execute time is: ", (end-start), "seconds")
    return y


# # convolution layer using for-loop and no padding
# def convolution_layer_for_loop_version(input_matrix, convolution_filter, strides, pad_size):
#     # input_matrix must be of shape: (sample_number, row, column, channel_number)
#     # since input_matrix has been padded, it's row and column will be larger than before.
#     # row = row + padding size * 2
#     # column = row + padding size * 2
#     # convolution_filter must be of shape: (n, n, channel_number, convolution_filter_channel_number)
#     # strides must be of shape: (sample_number_stride, row_stride, column_stride, channel_number_stride)
#     sample_number = input_matrix.shape[0]
#     row = input_matrix.shape[1]
#     column = input_matrix.shape[2]
#     channel_number = input_matrix.shape[3]
#     n = convolution_filter.shape[0]
#     convolution_filter_channel_number = convolution_filter.shape[3]
#     sample_number_stride = strides[0]
#     row_stride = strides[1]
#     column_stride = strides[2]
#     channel_number_stride = strides[3]
#     result = np.zeros((sample_number, int((row - n + 2*pad_size) / row_stride + 1),
#                        int((column - n + 2*pad_size) / column_stride+1), convolution_filter_channel_number))
#     for i in range(0, sample_number, sample_number_stride):
#         for j in range(0, row, row_stride):
#             for k in range(0, column, column_stride):
#                 for l in range(convolution_filter_channel_number):
#                     single_matrix = input_matrix[i, j:j+n, k:k+n, l:channel_number]
#                     result[int((i+1)/sample_number_stride-1), int((j+1)/row_stride-1), int((k+1)/column_stride-1), l]\
#                          = tf.multiply(single_matrix, convolution_filter[:, :, :, l])
#     return result


# pooling layer:
def pooling(input_matrix, pooling_size, strides, pool_way):
    # pooling_size: [1, height, width, 1]
    # strides: [1, stride, stride, 1]
    # print("Pooling...")
    # start = time.time()
    p_n = pooling_size[1]
    p_p = pooling_size[2]

    s_n = strides[1]
    s_p = strides[2]

    i_m = input_matrix.shape[0]
    i_n = input_matrix.shape[1]
    i_p = input_matrix.shape[2]
    i_q = input_matrix.shape[3]

    output_matrix = np.zeros((i_m, int(i_n/s_n), int(i_p/s_p), i_q))
    index_matrix = np.ones((i_m, i_n, i_p, i_q))
    # for i in range(i_m):
    #     for j in range(0, i_n, s_n):
    #         for k in range(0, i_p, s_p):
    #             for l in range(i_q):
    #                 a = input_matrix[i, j:j + s_n, k:k + s_p, l]
    #                 if pool_way == 'max_pool':
    #                     b = tf.reduce_max(a)
    #                     output_matrix[i, int(j / s_n), int(k / s_p), l] = b
    #                     c = np.argwhere(a == b)
    #                     d = np.zeros((s_n, s_p))
    #                     c0 = int(c[0][0])
    #                     c1 = int(c[0][1])
    #                     d[c0, c1] = 1
    #                     index_matrix[i, j:j+s_n, k:k+s_p, l] = d
    #                 elif pool_way == 'mean_pool':
    #                     output_matrix[i, int(j / s_n), int(k / s_p), l] = tf.reduce_mean(a)
    #                     index_matrix = np.ones((i_m, i_n, i_p, i_q))

    # using tensor_flow built-in functions:
    if pool_way == 'max_pool':
        output_matrix = tf.nn.max_pool(input_matrix, ksize=pooling_size, strides=strides, padding='VALID')
    elif pool_way == 'mean_pool':
        output_matrix = tf.nn.avg_pool(input_matrix, ksize=pooling_size, strides=strides, padding='VALID')
    # end = time.time()
    # print("Done.")
    # print("Execute time is: ", (end-start), "seconds")
    return output_matrix, index_matrix


# activation function:
def activate(input_matrix, function):
    # print("Activating...")
    # start = time.time()
    if function == 'tanh':
        output_matrix = tf.math.tanh(input_matrix)
    elif function == 'ReLU':
        output_matrix = tf.nn.relu(input_matrix)
    elif function == 'exponential':
        output_matrix = tf.subtract(tf.exp(input_matrix), 1)
    else:
        output_matrix = input_matrix
    # end = time.time()
    # print("Done.")
    # print("Execute time is: ", (end-start), "seconds")
    return output_matrix


# flatten:
def flatten(input_matrix):
    # print("Flatting...")
    # start = time.time()
    m = input_matrix.shape[0]
    n = input_matrix.shape[1]
    p = input_matrix.shape[2]
    q = input_matrix.shape[3]
    input_matrix = input_matrix.numpy()
    output_matrix = input_matrix.reshape((m, n*p*q))
    # end = time.time()
    # print("Done.")
    # print("Execute time is: ", (end-start), "seconds")
    return output_matrix


# # fully connected layer:
# # this layer does not need this, or it will be more complex in the backward propagation.
# def fully_connected_layer_single(input_matrix, activate_function, weights, bias):
#     a = activate(input_matrix, activate_function)
#     output_matrix = tf.add(tf.matmul(a, weights), bias)
#     return output_matrix


# soft_max function:
def soft_max(input_matrix):
    # print("Calculating soft max function...")
    # start = time.time()
    # print("INPUT_Matrix SIZE is: ", input_matrix.shape)
    h = tf.exp(input_matrix)
    # print("h size is:", tf.shape(h))
    denominator = tf.reduce_sum(h, axis=1, keepdims=True)
    hypothesis = tf.divide(h, denominator)
    # print("hypothesis size is", tf.shape(hypothesis))
    # end = time.time()
    # print("Done.")
    # print("Execute time is: ", (end-start), "seconds")
    return hypothesis


# forward propagation:
def forward_propagation(x, parameter):
    # print("##################FORWARD PROPAGATING#################")
    # start = time.time()
    # load required parameters:
    cv_ft1 = parameter["cv_ft1"]
    cv_s1 = parameter["cv_s1"]
    ac_fnc_cv = parameter["ac_fnc_cv"]
    po_sz1 = parameter["po_sz1"]
    po_str1 = parameter["po_str1"]
    po_wy = parameter['po_wy']
    cv_ft2 = parameter["cv_ft2"]
    cv_s2 = parameter["cv_s2"]
    po_sz2 = parameter["po_sz2"]
    po_str2 = parameter["po_str2"]
    ac_fnc_fl = parameter["ac_fnc_fl"]
    fl_w1 = parameter["fl_w1"]
    fl_b1 = parameter["fl_b1"]
    fl_w2 = parameter["fl_w2"]
    fl_b2 = parameter["fl_b2"]
    fl_w3 = parameter["fl_w3"]
    fl_b3 = parameter["fl_b3"]
    fl_w4 = parameter["fl_w4"]

    # convolution forward
    # a1_1 = padding(input_matrix=x, padding_size=1)
    z1_1 = convolution_layer_tf(input_matrix=x, convolution_filter=cv_ft1, strides=cv_s1)
    a1_2 = activate(input_matrix=z1_1, function=ac_fnc_cv)
    z1_2, idx_po1 = pooling(input_matrix=a1_2, pooling_size=po_sz1, strides=po_str1, pool_way=po_wy)

    # a2_1 = padding(input_matrix=z1_2, padding_size=1)
    z2_1 = convolution_layer_tf(input_matrix=z1_2, convolution_filter=cv_ft2, strides=cv_s2)
    a2_2 = activate(input_matrix=z2_1, function=ac_fnc_cv)
    z2_2, idx_po2 = pooling(input_matrix=a2_2, pooling_size=po_sz2, strides=po_str2, pool_way=po_wy)

    # flatten
    z2_2_flatten = flatten(z2_2)

    # # fully connected forward
    # fully connected 1st layer
    fl_a1 = activate(input_matrix=z2_2_flatten, function=ac_fnc_fl)
    fl_z1 = tf.add(tf.matmul(fl_a1, fl_w1), fl_b1)

    # fully connected 2nd layer
    fl_a2 = activate(input_matrix=fl_z1, function=ac_fnc_fl)
    fl_z2 = tf.add(tf.matmul(fl_a2, fl_w2), fl_b2)

    # fully connected 3rd layer
    fl_a3 = activate(input_matrix=fl_z2, function=ac_fnc_fl)
    fl_z3 = tf.add(tf.matmul(fl_a3, fl_w3), fl_b3)

    # the last fully connected layer does not need bias
    fl_a4 = activate(input_matrix=fl_z3, function=ac_fnc_fl)
    fl_z4 = tf.matmul(fl_a4, fl_w4)

    # soft_max output
    hypothesis = soft_max(fl_z4)

    # save the result:
    forward_result = {
        # "a1_1": a1_1,
        "z1_1": z1_1,
        "a1_2": a1_2,
        "z1_2": z1_2,
        # "a2_1": a2_1,
        "z2_1": z2_1,
        "a2_2": a2_2,
        "z2_2": z2_2,
        "fl_a1": fl_a1,
        "fl_z1": fl_z1,
        "fl_a2": fl_a2,
        "fl_z2": fl_z2,
        "fl_a3": fl_a3,
        "fl_z3": fl_z3,
        "fl_a4": fl_a4,
        "fl_z4": fl_z4,
        'idx_po1': idx_po1,
        'idx_po2': idx_po2,
        "hypothesis": hypothesis
    }
    # end = time.time()
    # print("Execute time is: ", (end-start), "seconds")
    # print("##################FORWARD PROPAGATION DONE#################")

    return forward_result


# define the cost function:
def compute_cost(hypothesis, y_train):
    # print("Computing loss...")
    # start = time.time()
    m = y_train.shape[0]
    j = -tf.reduce_sum(tf.multiply(y_train, tf.math.log(hypothesis)))
    j = tf.divide(j, m)

    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y_train))

    # end = time.time()
    # print("Done.")
    # print("Execute time is: ", (end-start), "seconds")
    return j


# define the activation function's derivative.
def activation_derivative(input_matrix, activation):
    # print("Calculating Activation Derivative...")
    # start = time.time()
    if activation == 'tanh':
        output_matrix = tf.subtract(1, tf.pow(input_matrix, 2))
    elif activation == 'ReLU':
        output_matrix = tf.cast(input_matrix > 0, tf.float64)
    elif activation == 'exponential':
        output_matrix = tf.add(input_matrix, 1)
    else:
        output_matrix = tf.ones(input_matrix.shape, dtype=tf.float64)
    # end = time.time()
    # print("Done.")
    # print("Execute time is: ", (end-start), "seconds")
    return output_matrix


# Max/Avg Pool back.
def pool_back(input_matrix, pooling_size, strides, pool_way, index_matrix):
    # print("Pooling Back...")
    # start = time.time()
    p_n = pooling_size[1]
    p_p = pooling_size[2]

    s_n = strides[1]
    s_p = strides[2]

    i_m = input_matrix.shape[0]
    i_n = input_matrix.shape[1]
    i_p = input_matrix.shape[2]
    i_q = input_matrix.shape[3]

    output_matrix = np.zeros((i_m, i_n*s_n, i_p*s_p, i_q))

    b = input_matrix.numpy()
    b = b.reshape((i_m, i_n*i_p, 1, i_q))
    b = tf.tile(b, [1, 1, s_p, 1])
    b = b.numpy()
    b = b.reshape((i_m, i_n, i_p*s_p, i_q))
    b = tf.transpose(b, [0, 2, 1, 3])
    b = b.numpy()
    b = b.reshape((i_m, i_n*i_p*s_p, 1, i_q))
    b = tf.tile(b, [1, 1, s_n, 1])
    b = b.numpy()
    b = b.reshape((i_m, i_p*s_p, i_n*s_n, i_q))
    b = tf.transpose(b, [0, 2, 1, 3])
    output_matrix = b

    if pool_way == 'max_pool':
        output_matrix = tf.multiply(output_matrix, index_matrix)
    elif pool_way == 'mean_pool':
        output_matrix = tf.divide(output_matrix, p_n*p_p)
    # end = time.time()
    # print("Done.")
    # print("Execute time is: ", (end-start), "seconds")
    return output_matrix


# convolution layer backward propagation:
def cv_back(input_matrix, cv_frw_input, cv_ft, strides):
    # strides is a list: [sample_number_stride, row_pixel_stride, column_pixel_stride, channel_stride]
    # padding_size is a scalar.
    # print("Calculating Convolution Backward Propagation...")
    # start = time.time()

    s = strides[1]
    m = cv_frw_input.shape[0]
    for_da_input_matrix = padding(input_matrix=input_matrix, padding_size=s)
    # for_da_input_matrix = for_da_input_matrix.numpy()
    # input_matrix = input_matrix.numpy()
    # cv_ft = cv_ft.numpy()
    flipped_cv_frw_in = np.flip(np.flip(cv_frw_input, axis=1), axis=2)
    transposed_cv_frw_in = tf.transpose(flipped_cv_frw_in, [3, 1, 2, 0])
    for_dcv_ft_in = tf.transpose(input_matrix, [1, 2, 0, 3])
    da = tf.nn.conv2d(for_da_input_matrix, np.flip(np.flip(cv_ft, axis=0), axis=1), strides=strides, padding='VALID')
    d_cv_ft = tf.nn.conv2d(
        transposed_cv_frw_in, for_dcv_ft_in, strides=strides, padding='VALID')
    d_cv_ft = tf.transpose(d_cv_ft, [1, 2, 0, 3])
    d_cv_ft = tf.divide(d_cv_ft, m)

    # # using tensor flow built-in functions.
    # input_sizes = cv_frw_input.shape
    # filter_sizes = cv_ft.shape
    # da = tf.nn.depthwise_conv2d_backprop_input(input_sizes, cv_ft, input_matrix, strides, padding='SAME')
    # d_cv_ft= tf.nn.depthwise_conv2d_backprop_filter(cv_frw_input, filter_sizes, input_matrix, strides, padding='SAME')

    # end = time.time()
    # print("Done.")
    # print("Execute time is: ", (end-start), "seconds")
    return da, d_cv_ft


# padding layer backward propagation:
def pad_back(input_matrix, strides):
    # input_matrix is of shape: (m, row_pixel, column_pixel, channel)
    # strides is a list: [m_stride, row_stride, column_stride, channel_stride)
    # print("Padding back...")
    # start = time.time()
    row_stride = strides[1]
    column_stride = strides[2]
    output_matrix = input_matrix.numpy()
    for i in range(row_stride):
        output_matrix = np.delete(output_matrix, i, axis=1)
        output_matrix = np.delete(output_matrix, -1-i, axis=1)
    for i in range(column_stride):
        output_matrix = np.delete(output_matrix, i, axis=2)
        output_matrix = np.delete(output_matrix, -1-i, axis=2)
    # end = time.time()
    # print("Done.")
    # print("Execute time is: ", (end-start), "seconds")
    return output_matrix


# random mini batches.
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


# backward propagation:
def backward_propagation(x, y, forward_result, parameter):
    # load required data:
    # print("###################BACKWARD PROPAGATION#####################")
    # start = time.time()
    m = y.shape[0]
    hypothesis = forward_result['hypothesis']
    fl_a4 = forward_result['fl_a4']
    fl_w4 = parameter['fl_w4']
    ac_fnc_fl = parameter['ac_fnc_fl']
    fl_a3 = forward_result['fl_a3']
    fl_w3 = parameter['fl_w3']
    fl_a2 = forward_result['fl_a2']
    fl_w2 = parameter['fl_w2']
    fl_a1 = forward_result['fl_a1']
    fl_w1 = parameter['fl_w1']
    z1_2 = forward_result['z1_2']
    z2_2 = forward_result['z2_2']
    po_wy = parameter['po_wy']
    po_sz2 = parameter["po_sz2"]
    po_str2 = parameter["po_str2"]
    a2_2 = forward_result['a2_2']
    ac_fnc_cv = parameter['ac_fnc_cv']
    cv_ft2 = parameter["cv_ft2"]
    cv_s2 = parameter["cv_s2"]
    cv_ft1 = parameter["cv_ft1"]
    cv_s1 = parameter["cv_s1"]
    po_sz1 = parameter["po_sz1"]
    po_str1 = parameter["po_str1"]
    a1_2 = forward_result['a1_2']
    idx_po1 = forward_result['idx_po1']
    idx_po2 = forward_result['idx_po2']

    # backward propagate steps:
    # soft_max and the last fully connected layer backward step:
    d_fl_z4 = tf.subtract(hypothesis, y)
    d_fl_w4 = tf.divide(tf.matmul(tf.transpose(fl_a4), d_fl_z4), m)
    d_fl_a4 = tf.matmul(d_fl_z4, tf.transpose(fl_w4))

    # the 3rd fully connected layer backward step:
    d_fl_z3 = tf.multiply(d_fl_a4, activation_derivative(input_matrix=fl_a4, activation=ac_fnc_fl))
    d_fl_w3 = tf.divide(tf.matmul(tf.transpose(fl_a3), d_fl_z3), m)
    d_fl_b3 = tf.divide(tf.reduce_sum(d_fl_z3, axis=0), m)
    d_fl_a3 = tf.matmul(d_fl_z3, tf.transpose(fl_w3))

    # the 2nd fully connected layer backward step:
    d_fl_z2 = tf.multiply(d_fl_a3, activation_derivative(input_matrix=fl_a3, activation=ac_fnc_fl))
    d_fl_w2 = tf.divide(tf.matmul(tf.transpose(fl_a2), d_fl_z2), m)
    d_fl_b2 = tf.divide(tf.reduce_sum(d_fl_z2, axis=0), m)
    d_fl_a2 = tf.matmul(d_fl_z2, tf.transpose(fl_w2))

    # the 1st fully connected layer backward step:
    d_fl_z1 = tf.multiply(d_fl_a2, activation_derivative(input_matrix=fl_a2, activation=ac_fnc_fl))
    d_fl_w1 = tf.divide(tf.matmul(tf.transpose(fl_a1), d_fl_z1), m)
    d_fl_b1 = tf.divide(tf.reduce_sum(d_fl_z1, axis=0), m)
    d_fl_a1 = tf.matmul(d_fl_z1, tf.transpose(fl_w1))
    d_z2_2 = tf.multiply(d_fl_a1, activation_derivative(input_matrix=fl_a1, activation=ac_fnc_fl))

    # reshape back to origin dimension.
    d_z2_2 = tf.reshape(d_z2_2, z2_2.shape)

    # Max/Avg Pooling backward propagation.
    d_a2_2 = pool_back(input_matrix=d_z2_2, pooling_size=po_sz2, strides=po_str2, pool_way=po_wy, index_matrix=idx_po2)

    # activation function backward propagation.
    d_z2_1 = tf.multiply(d_a2_2, activation_derivative(input_matrix=a2_2, activation=ac_fnc_cv))

    # convolution layer backward propagation.
    d_a2_1, d_cv_ft2 = cv_back(d_z2_1, z1_2, cv_ft=cv_ft2, strides=cv_s2)

    # pad layer backward propagation.
    # d_z1_2 = pad_back(input_matrix=d_a2_1, strides=cv_s2)

    # Max/Avg Pooling backward propagation.
    d_a1_2 = pool_back(input_matrix=d_a2_1, pooling_size=po_sz1, strides=po_str1, pool_way=po_wy, index_matrix=idx_po1)

    # activation function backward propagation.
    d_z1_1 = tf.multiply(d_a1_2, activation_derivative(input_matrix=a1_2, activation=ac_fnc_cv))

    # convolution layer backward propagation.
    d_a1_1, d_cv_ft1 = cv_back(d_z1_1, x, cv_ft=cv_ft1, strides=cv_s1)

    # pad layer backward propagation.
    # d_x = pad_back(input_matrix=d_a1_1, strides=cv_s1)

    # # save the grads.
    # d_fl_w4 = d_fl_w4.numpy()
    # np.savetxt('dataset/d_f1_w4.txt', d_fl_w4)
    # d_fl_w3 = d_fl_w3.numpy()
    # np.savetxt('dataset/d_fl_w3.txt', d_fl_w3)
    # d_fl_b3 = d_fl_b3.numpy()
    # np.savetxt('dataset/d_fl_b3.txt', d_fl_b3)
    # d_fl_w2 = d_fl_w2.numpy()
    # np.savetxt('dataset/d_fl_w2.txt', d_fl_w2)
    # d_fl_b2 = d_fl_b2.numpy()
    # np.savetxt('dataset/d_fl_b2.txt', d_fl_b2)
    # d_fl_w1 = d_fl_w1.numpy()
    # np.savetxt('dataset/d_fl_w1.txt', d_fl_w1)
    # d_fl_b1 = d_fl_b1.numpy()
    # np.savetxt('dataset/d_fl_b1.txt', d_fl_b1)
    # d_cv_ft2 = d_cv_ft2.numpy()
    # d_cv_ft2 = d_cv_ft2.reshape((3, 3))
    # np.savetxt('dataset/d_cv_ft2.txt', d_cv_ft2)
    # d_cv_ft1 = d_cv_ft1.numpy()
    # d_cv_ft1 = d_cv_ft1.reshape((3, 3))
    # np.savetxt('dataset/d_cv_ft1.txt', d_cv_ft1)

    grad = {
        'd_fl_w4': d_fl_w4,
        'd_fl_w3': d_fl_w3,
        'd_fl_b3': d_fl_b3,
        'd_fl_w2': d_fl_w2,
        'd_fl_b2': d_fl_b2,
        'd_fl_w1': d_fl_w1,
        'd_fl_b1': d_fl_b1,
        'd_cv_ft2': d_cv_ft2,
        'd_cv_ft1': d_cv_ft1
    }

    # end = time.time()
    # print("Done.")
    # print("Execute time is: ", (end-start), "seconds")
    # print("#######################BACKWARD PROPAGATION DONE#####################")

    return grad


# update parameter:
def update_parameter(parameter, grad, learning_rate):
    # load data:
    # print("Updating parameter...")
    # start = time.time()
    fl_w1 = parameter["fl_w1"]
    fl_b1 = parameter["fl_b1"]
    fl_w2 = parameter["fl_w2"]
    fl_b2 = parameter["fl_b2"]
    fl_w3 = parameter["fl_w3"]
    fl_b3 = parameter["fl_b3"]
    fl_w4 = parameter["fl_w4"]
    cv_ft1 = parameter['cv_ft1']
    cv_ft2 = parameter['cv_ft2']

    # d_fl_w4 = np.loadtxt('dataset/d_fl_w4.txt')
    # d_fl_w3 = np.loadtxt('dataset/d_fl_w3.txt')
    # d_fl_b3 = np.loadtxt('dataset/d_fl_b3.txt')
    # d_fl_w2 = np.loadtxt('dataset/d_fl_w2.txt')
    # d_fl_b2 = np.loadtxt('dataset/d_fl_b2.txt')
    # d_fl_w1 = np.loadtxt('dataset/d_fl_w1.txt')
    # d_fl_b1 = np.loadtxt('dataset/d_fl_b1.txt')
    # d_cv_ft1 = np.loadtxt('dataset/d_cv_ft1.txt')
    # d_cv_ft1 = d_cv_ft1.reshape((3, 3, 1, 1))
    # d_cv_ft2 = np.loadtxt('dataset/d_cv_ft2.txt')
    # d_cv_ft2 = d_cv_ft2.reshape((3, 3, 1, 1))

    d_fl_w4 = grad['d_fl_w4']
    d_fl_w3 = grad['d_fl_w3']
    d_fl_b3 = grad['d_fl_b3']
    d_fl_w2 = grad['d_fl_w2']
    d_fl_b2 = grad['d_fl_b2']
    d_fl_w1 = grad['d_fl_w1']
    d_fl_b1 = grad['d_fl_b1']
    d_cv_ft1 = grad['d_cv_ft1']
    d_cv_ft2 = grad['d_cv_ft2']

    fl_w1 -= learning_rate * d_fl_w1
    fl_b1 -= learning_rate * d_fl_b1
    fl_w2 -= learning_rate * d_fl_w2
    fl_b2 -= learning_rate * d_fl_b2
    fl_w3 -= learning_rate * d_fl_w3
    fl_b3 -= learning_rate * d_fl_b3
    fl_w4 -= learning_rate * d_fl_w4
    cv_ft1 -= learning_rate * d_cv_ft1
    cv_ft2 -= learning_rate * d_cv_ft2

    parameter['fl_w1'] = fl_w1.numpy()
    parameter['fl_b1'] = fl_b1.numpy()
    parameter['fl_w2'] = fl_w2.numpy()
    parameter['fl_b2'] = fl_b2.numpy()
    parameter['fl_w3'] = fl_w3.numpy()
    parameter['fl_b3'] = fl_b3.numpy()
    parameter['fl_w4'] = fl_w4.numpy()
    parameter['cv_ft1'] = cv_ft1.numpy()
    parameter['cv_ft2'] = cv_ft2.numpy()

    # end = time.time()
    # print("Done.")
    # print("Execute time is: ", (end-start), "seconds")

    return parameter


# accuracy rate calculate:
def compute_accuracy(hypothesis, y):
    # print("Computing accuracy...")
    # start = time.time()
    m = y.shape[0]
    # print("hypothesis size is:", tf.shape(hypothesis))
    hypothesis = hypothesis.numpy()
    # print("numpy hypothesis size is:", hypothesis.shape)
    h = np.argmax(hypothesis, axis=1)
    # print("h size is:", h.shape)
    train_accuracy = np.sum(((h - y) == 0) + 0) / m
    # end = time.time()
    # print("Done.")
    # print("Execute time is: ", (end-start), "seconds")
    return train_accuracy


# define the cnn model:
def model_cnn(x_train, y_train, x_test, parameter, epoch=200, learning_rate=0.5, mini_batch_size=1024):

    m = y_train.shape[0]
    seed = 0
    loss = 0
    accuracy = 0
    loss_list = []
    epoch_list = []
    accuracy_list = []
    forward_result = {}

    for i in range(epoch):
        print("#########################EPOCH ", i+1, "#############################")
        epoch_list.append(i+1)
        # mini_batch_cost = 0.
        # num_mini_batches = int(m / mini_batch_size)  # number of mini batches of size mini batch_size in the train set
        seed = seed + 1
        mini_batches = random_mini_batches(x_train, y_train, mini_batch_size, seed)
        j = 0

        for mini_batch in mini_batches:
            j = j+1
            # print("---------------------mini-batch-", j, '--------------------')
            # Select a mini batch
            (mini_batch_x, mini_batch_y) = mini_batch
            # mini_batch_y_orig = np.argmax(mini_batch_y, axis=1)
            forward_result = forward_propagation(mini_batch_x, parameter)
            # hypothesis = forward_result['hypothesis']
            # loss = compute_cost(hypothesis=hypothesis, y_train=mini_batch_y)
            grad = backward_propagation(mini_batch_x, mini_batch_y, forward_result=forward_result, parameter=parameter)
            parameter = update_parameter(parameter=parameter, grad=grad, learning_rate=learning_rate)
            # accuracy = compute_accuracy(hypothesis, mini_batch_y_orig)
            # print("Mini-batch %d: loss=%f\n train accuracy is: %f" % (j + 1, loss, accuracy))

        mini_batches_test = random_mini_batches(x_train, y_train, mini_batch_size=16384, seed=seed)
        (mini_batch_x_test, mini_batch_y_test) = mini_batches_test[0]
        mini_batch_y_test_orig = np.argmax(mini_batch_y_test, axis=1)
        forward_result = forward_propagation(mini_batch_x_test, parameter)
        hypothesis = forward_result['hypothesis']
        loss = compute_cost(hypothesis=hypothesis, y_train=mini_batch_y_test)
        grad = backward_propagation(
            mini_batch_x_test, mini_batch_y_test, forward_result=forward_result, parameter=parameter)
        parameter = update_parameter(parameter=parameter, grad=grad, learning_rate=learning_rate)
        accuracy = compute_accuracy(hypothesis, mini_batch_y_test_orig)
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        print("Epoch %d: loss=%f\n train accuracy is: %f" % (i+1, loss, accuracy))

        if (i+1) == 10:
            learning_rate = learning_rate/10

        if ((i+1) % 50) == 0:
            plt.plot(epoch_list, loss_list)
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.show()
            plt.plot(epoch_list, accuracy_list)
            plt.xlabel("epoch")
            plt.ylabel("accuracy")
            plt.show()

    forward_result = forward_propagation(x_test, parameter)
    hypothesis = forward_result['hypothesis']
    hypothesis = hypothesis.numpy()
    h_test = np.argmax(hypothesis, axis=1).reshape(x_test.shape[0], 1)
    n_test = np.array(range(1, x_test.shape[0] + 1)).reshape(x_test.shape[0], 1)
    h_test = np.hstack((n_test, h_test))
    df_h_test = pd.DataFrame(h_test, columns=["ImageId", "Label"])
    df_h_test.to_csv("dataset/predictions.csv", index=False)



