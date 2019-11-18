from cnn_functions import *
from keras.utils import to_categorical

X_train_flatten, Y_train_orig = load_data("dataset/train.csv")
X_test_flatten = np.array(pd.read_csv("dataset/test.csv"))

X_train_orig = X_train_flatten.reshape((42000, 28, 28, 1))
X_test_orig = X_test_flatten.reshape((28000, 28, 28, 1))

X_train = X_train_orig/255
X_test = X_test_orig/255

Y_train = to_categorical(Y_train_orig)

# # cast the integer value to tf.float64
# X_train = tf.cast(X_train, tf.float64)
# X_test = tf.cast(X_test, tf.float64)
# Y_train = tf.cast(Y_train, tf.float64)

# convolution_filter1 = np.random.random((3, 3, 1, 1))
# convolution_filter2 = np.random.random((3, 3, 1, 1))
# strides = [1, 2, 2, 1]
# padding_size = 1

cv_ft1 = np.random.randn(3, 3, 1, 1)*255
cv_s1 = [1, 1, 1, 1]
ac_fnc_cv = 'ReLU'
po_sz1 = [1, 2, 2, 1]
po_str1 = [1, 2, 2, 1]
po_wy = 'mean_pool'
cv_ft2 = np.random.randn(3, 3, 1, 1)*255
cv_s2 = [1, 1, 1, 1]
po_sz2 = [1, 2, 2, 1]
po_str2 = [1, 2, 2, 1]
ac_fnc_fl = 'ReLU'
fl_w1 = np.random.randn(49, 64)
fl_w1 = fl_w1*0.01
fl_b1 = np.random.randn(1, 64)
fl_w2 = np.random.randn(64, 32)
fl_w2 = fl_w2*0.01
fl_b2 = np.random.randn(1, 32)
fl_w3 = np.random.randn(32, 16)
fl_w3 = fl_w3*0.01
fl_b3 = np.random.randn(1, 16)
fl_w4 = np.random.randn(16, 10)
fl_w4 = fl_w4*0.01

# load the parameters
fl_w4 = np.loadtxt('dataset/fl_w4.txt')
fl_w3 = np.loadtxt('dataset/fl_w3.txt')
fl_b3 = np.loadtxt('dataset/fl_b3.txt')
fl_w2 = np.loadtxt('dataset/fl_w2.txt')
fl_b2 = np.loadtxt('dataset/fl_b2.txt')
fl_w1 = np.loadtxt('dataset/fl_w1.txt')
fl_b1 = np.loadtxt('dataset/fl_b1.txt')
cv_ft1 = np.loadtxt('dataset/cv_ft1.txt')
cv_ft1 = cv_ft1.reshape((3, 3, 1, 1))
cv_ft2 = np.loadtxt('dataset/cv_ft2.txt')
cv_ft2 = cv_ft2.reshape((3, 3, 1, 1))

parameter = {
    'cv_ft1': cv_ft1,
    'cv_s1': cv_s1,
    'ac_fnc_cv': ac_fnc_cv,
    'po_sz1': po_sz1,
    'po_str1': po_str1,
    'po_wy': po_wy,
    'cv_ft2': cv_ft2,
    'cv_s2': cv_s2,
    'po_sz2': po_sz2,
    'po_str2': po_str2,
    'ac_fnc_fl': ac_fnc_fl,
    'fl_w1': fl_w1,
    'fl_b1': fl_b1,
    'fl_w2': fl_w2,
    'fl_b2': fl_b2,
    'fl_w3': fl_w3,
    'fl_b3': fl_b3,
    'fl_w4': fl_w4
}

model_cnn(x_train=X_train, y_train=Y_train, x_test=X_test, parameter=parameter,
          epoch=50, learning_rate=0.001, mini_batch_size=512)

