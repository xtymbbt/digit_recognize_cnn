from PIL import Image

from cnn_functions import *

X_train_flatten, Y_train_orig = load_data("dataset/train.csv")
X_test_flatten = np.array(pd.read_csv("dataset/test.csv"))

X_train_orig = X_train_flatten.reshape((42000, 28, 28, 1))
X_test_orig = X_test_flatten.reshape((28000, 28, 28, 1))

X_train = X_train_orig/255
X_test = X_test_orig/255

cv_s1 = [1, 1, 1, 1]
ac_fnc_cv = 'ReLU'
po_sz1 = [1, 2, 2, 1]
po_str1 = [1, 2, 2, 1]
po_wy = 'mean_pool'
cv_s2 = [1, 1, 1, 1]
po_sz2 = [1, 2, 2, 1]
po_str2 = [1, 2, 2, 1]
ac_fnc_fl = 'ReLU'

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
loop = True
while loop:
    seed_train = int(np.random.random(1) * 42000)
    seed_test = int(np.random.random(1) * 28000)

    test_img = X_train[seed_train, :, :, :]
    show = test_img.reshape((test_img.shape[0], test_img.shape[1]))
    plt.imshow((show * 255).astype(np.uint8), cmap=plt.cm.gray, interpolation='nearest')
    plt.show()
    test_img = test_img.reshape((1, test_img.shape[0], test_img.shape[1], test_img.shape[2]))

    forward_result = forward_propagation(test_img, parameter)

    hypothesis = forward_result['hypothesis']
    h = np.argmax(hypothesis)
    print("The digit in this image is: " + str(h))
    input_ = input("Input \"exit\" to exit the while loop.")
    if input_ == "exit":
        loop = False


