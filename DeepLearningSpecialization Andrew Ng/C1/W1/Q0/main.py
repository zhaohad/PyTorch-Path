import numpy as np


def get_numpy(*a):
    n = 1
    for i in a:
        n *= i
    return (np.array(range(n)) + 1).reshape(a)


# a * b 与 b * a 一致，与numpy.multiply(a, b)一致
a = get_numpy(1, 4)
b = 2
print(f"a = {a}, a.shape = {a.shape}")
# print(f"b = {b}, a.shape = {b.shape}") # int没有shape
print(f"a * b = {a * b}, (a * b).shape = {(a * b).shape}")
print("\n\n")


a = get_numpy(1, 4)
b = get_numpy(2)
print(f"a = {a}, a.shape = {a.shape}")
print(f"b = {b}, a.shape = {b.shape}")
# print(f"a * b = {a * b}, (a * b).shape = {(a * b).shape}") # 不可相乘
print("\n\n")

a = get_numpy(1, 4)
b = get_numpy(1, 4)
print(f"a = {a}, a.shape = {a.shape}")
print(f"b = {b}, a.shape = {b.shape}")
print(f"a * b = {a * b}, (a * b).shape = {(a * b).shape}")
print("\n\n")

a = get_numpy(1, 4)
b = get_numpy(4)  # 默认是1行n列的数组
print(f"a = {a}, a.shape = {a.shape}")
print(f"b = {b}, a.shape = {b.shape}")
print(f"a * b = {a * b}, (a * b).shape = {(a * b).shape}")
print("\n\n")

a = get_numpy(1, 4)
b = get_numpy(1, 4)
print(f"a = {a}, a.shape = {a.shape}")
print(f"b = {b}, a.shape = {b.shape}")
print(f"a * b = {a * b}, (a * b).shape = {(a * b).shape}")
print("\n\n")

a = get_numpy(1, 4)
b = get_numpy(4, 1)
print(f"a = {a}, a.shape = {a.shape}")
print(f"b = {b}, a.shape = {b.shape}")
print(f"a * b = {a * b}, (a * b).shape = {(a * b).shape}")
print("\n\n")

get_numpy(1, 4)
get_numpy(2, 2)
print(f"a = {a}, a.shape = {a.shape}")
print(f"b = {b}, a.shape = {b.shape}")
# print(f"a * b = {a * b}, (a * b).shape = {(a * b).shape}") # 不可相乘
print("\n\n")

a = get_numpy(2, 3, 1)
b = get_numpy(1, 2)
print(f"a = {a}, a.shape = {a.shape}")
print(f"b = {b}, a.shape = {b.shape}")
print(f"a * b = {a * b}, (a * b).shape = {(a * b).shape}")
print("\n\n")

a = get_numpy(3, 2, 1)
b = get_numpy(2, 3)
print(f"a = {a}, a.shape = {a.shape}")
print(f"b = {b}, a.shape = {b.shape}")
print(f"a * b = {a * b}, (a * b).shape = {(a * b).shape}")
print("\n\n")

a = get_numpy(1, 4)
b = get_numpy(2, 1)
print(f"a = {a}, a.shape = {a.shape}")
print(f"b = {b}, a.shape = {b.shape}")
print(f"a * b = {a * b}, (a * b).shape = {(a * b).shape}")
print("\n\n")

a = get_numpy(1, 4)
b = get_numpy(4, 1)
print(f"a = {a}, a.shape = {a.shape}")
print(f"b = {b}, a.shape = {b.shape}")
print(f"np.dot(a, b) = {np.dot(a, b)}, np.dot(a, b).shape = {np.dot(a, b).shape}")
print(f"np.dot(b, a) = {np.dot(b, a)}, np.dot(b, a).shape = {np.dot(b, a).shape}")
print("\n\n")

a = get_numpy(2, 1, 4, 5)
b = get_numpy(1, 1, 5, 4)
print(f"a = {a}, a.shape = {a.shape}")
print(f"b = {b}, a.shape = {b.shape}")
print(f"np.dot(a, b) = {np.dot(a, b)}, np.dot(a, b).shape = {np.dot(a, b).shape}")
print(f"np.dot(b, a) = {np.dot(b, a)}, np.dot(b, a).shape = {np.dot(b, a).shape}")
print("\n\n")

a = get_numpy(2, 1, 4, 5)
b = get_numpy(1, 1, 5, 4)
print(f"a = {a}, a.shape = {a.shape}")
print(f"b = {b}, a.shape = {b.shape}")
print(f"np.dot(a, b) = {np.dot(a, b)}, np.dot(a, b).shape = {np.dot(a, b).shape}")
print(f"np.dot(b, a) = {np.dot(b, a)}, np.dot(b, a).shape = {np.dot(b, a).shape}")
print("\n\n")

a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
b = np.array([[[8, 7], [6, 5]], [[4, 3], [2, 1]]])

print(a.shape)
print(np.dot(a, b))
print(np.dot(a, b).shape)


A = 7
B = 6
C = 5
D = 4
E = 3
F = 2
G = 4
H = 1

a = get_numpy(A, B, C, D)
b = get_numpy(E, F, G, H)
print(f"np.dot(a, b) = {np.dot(a, b)}, np.dot(a, b).shape = {np.dot(a, b).shape}")
print("\n\n")