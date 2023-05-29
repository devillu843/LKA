import numpy as np

a = np.arange(30)
a = a.reshape(3, 5, 2)
print(a)

print(np.mean(a, axis=(0, 1)))
