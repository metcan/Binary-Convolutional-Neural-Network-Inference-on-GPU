import numpy as np

a = np.random.randint(0,100,[10,10])

print(a)

np.save('a.npy', a)