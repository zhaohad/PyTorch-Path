import numpy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm


def draw_3d(X, Y, Z):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z, vmin=Z.min() * 2, cmap=cm.Blues)
    plt.show()


def linear(w, b, x):
    return w.T.dot(x) + b


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def loss(yhat, y):
    return -(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))


def infer(img_path, W, B):
    img = Image.open(img_path)
    np_img = numpy.array(img, dtype=np.float32).transpose(2, 1, 0)
    np_img = normalize_image(np_img.reshape(-1, 1))
    y_hat = sigmoid(linear(W, B, np_img))
    return y_hat


def normalize_image(img):
    return img * 2 / 0xff - 1


# 读取图片
img = Image.open("../../../Data/630_422/train/cat0.png")
img.show()

X = numpy.array(img, dtype=np.float32).transpose(2, 1, 0)
SHOW_X = X[0]
# print(f"X.shape = {X.shape}, SHOW_X.shape = {SHOW_X.shape}")

AX_X = np.arange(0, SHOW_X.shape[0], 1)
AX_Y = np.arange(0, SHOW_X.shape[1], 1)
# print(f"AX_X = {AX_X} AX_X.shape = {AX_X.shape}")
# print(f"AX_Y = {AX_Y} AX_Y.shape = {AX_Y.shape}")
AX_X, AX_Y = np.meshgrid(AX_Y, AX_X)
AX_Z = -SHOW_X
draw_3d(AX_X, AX_Y, AX_Z)

learning_rate = 0.000005
epochs = 100

# 归一化，RGB取值限制到-1.0 ~ 1.0
AX_Z = -normalize_image(SHOW_X)
draw_3d(AX_X, AX_Y, AX_Z)

X = normalize_image(X.reshape(-1, 1))
W = np.zeros(X.shape)
B = np.zeros((1, 1))
Y = np.ones((1, 1))

for i in range(epochs):
    Z = linear(W, B, X)
    # print(f"Z = {Z}")
    # print(f"W = {W}")

    a = sigmoid(Z)

    L = loss(a, 1)
    print(f"epoch: {i}, a = {a}, Loss = {L}, Z = {Z}, B = {B}")

    if i == epochs - 1:
        break

    dz = a - Y
    dw = X * dz
    db = dz
    print(f"dz = {dz}, dw = {dw}, db = {db}")

    W = W - learning_rate * dw
    B = B - learning_rate * db


y_hat_cat0 = infer("../../../Data/630_422/train/cat0.png", W, B)
l = loss(y_hat_cat0, 1)
print(f"y_hat_cat0 = {y_hat_cat0}, loss = {l}")

y_hat_cat1 = infer("../../../Data/630_422/train/cat1.jpg", W, B)
l = loss(y_hat_cat1, 1)
print(f"y_hat_cat0 = {y_hat_cat1}, loss = {l}")
