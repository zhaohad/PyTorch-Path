import numpy
import numpy as np
from PIL import Image


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