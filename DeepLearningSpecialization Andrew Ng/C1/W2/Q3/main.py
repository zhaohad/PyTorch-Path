import numpy
import numpy as np
from PIL import Image
import func
import os

EPOCHS = 1000
LEARNING_RATE = 0.00005
CHANNEL = 3
WIDTH = 630
HEIGHT = 422


def normalize_image(img):
    return img * 2 / 0xff - 1


def read_train_data():
    dir_path = "../../../Data/630_422/train"
    imgs = []
    for root, _, files in os.walk(dir_path):
        imgs.extend(os.path.join(root, f) for f in files)
    imgs.sort()

    y = []
    for img in imgs:
        y.append(1 if img.__contains__('cat') else 0)

    Y = numpy.array(y).reshape((-1, 1))

    x = [numpy.array(Image.open(img), dtype=np.float32).transpose(2, 1, 0) for img in imgs]
    for i, e in enumerate(x):
        assert e.shape == (CHANNEL, WIDTH, HEIGHT)

    X = numpy.array([t.reshape(-1) for t in x])
    X = normalize_image(X)

    # X = X[0].reshape(1, -1)  # 只用第一张图
    # Y = Y[0].reshape(-1, 1)  # 只用第一张图

    return X, Y


def test(w, b):
    dir_path = "../../../Data/630_422/test"
    imgs = []
    for root, _, files in os.walk(dir_path):
        imgs.extend(os.path.join(root, f) for f in files)
    imgs.sort()

    for img in imgs:
        np_img = numpy.array(Image.open(img), dtype=np.float32).transpose(2, 1, 0)
        assert np_img.shape == (CHANNEL, WIDTH, HEIGHT)
        np_img = normalize_image(np_img.reshape(-1, 1))
        y_hat = func.sigmoid(func.linear(w, b, np_img))
        print(f"img: {img}, y_hat: {y_hat}")


def main():
    X, Y = read_train_data()
    m = X.shape[0]
    W = np.zeros((X.shape[1], 1))
    B = np.zeros((1, 1))
    J = np.zeros(1)
    print(f"X.shape = {X.shape}, Y.shape = {Y.shape}, W.shape = {W.shape}, B.shape = {B.shape}")

    for epoch in range(EPOCHS):
        Z = func.linear(w=W, b=B, x=np.expand_dims(X, axis=-1)).reshape(-1, 1)
        A = func.sigmoid(Z)
        dz = A - Y
        db = dz.sum() / m
        # print(f"X.shape = {X.shape}, dz.shape = {dz.shape}, Z.shape = {Z.shape}, A.shape = {A.shape}, Y.shape = {Y.shape}, X.T.shape = {X.T.shape}")
        dw = (X.T.dot(dz)) / m
        l = func.loss(A, Y)
        # print(f"A = {A}, loss = {l}, Z = {Z}")
        J = l.sum() / m
        # print(f"Z = {Z}, a = {A}, dz = {dz}")
        # print(f"epoch = {epoch} Z.shape = {Z.shape} A.shape = {A.shape} dz.shape = {dz.shape} l.shape = {l.shape} J = {J} db = {db} dw.shape = {dw.shape}")
        print(f"epoch = {epoch} J = {J}")
        if epoch == EPOCHS - 1:
            break

        # print(f"dz = {dz}, dw = {dw}, db = {db}, X = {X}")
        W = W - LEARNING_RATE * dw
        B = B - LEARNING_RATE * db

    test(w=W, b=B)


if __name__ == "__main__":
    main()