import numpy as np

x=np.random.randn(4,5)
r=np.sum(x**2)
print(r)
y=np.sqrt(r-x)
print(y)