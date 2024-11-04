import numpy as np
a=np.array([1, 3, 5, 2, 6, 9])
b=np.array([9, 5, 3, 2])
print(np.where(np.isin(a, b))[0])
