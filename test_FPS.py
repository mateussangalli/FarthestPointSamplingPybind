import FPS
import numpy as np
import time

print('testing using a 100 x 100 matrix') 
t0 = time.time()
a = np.random.rand(100, 100)

b = FPS.farthest_point_sampling(a, 30)

print(f'output_shape: {b.shape}')
print('indices:')
print(b)
print(f'elapsed time: {time.time() - t0}')


print('testing using a 10000 x 10000 matrix') 
t0 = time.time()
a = np.random.rand(10000, 10000)

b = FPS.farthest_point_sampling(a, 1000)

print(f'output_shape: {b.shape}')
print(f'elapsed time: {time.time() - t0}')


print('testing using 10 10000 x 10000 matrix') 
t0 = time.time()
for i in range(10):
    a = np.random.rand(10000, 10000)
    b = FPS.farthest_point_sampling(a, 1000)
print(f'elapsed time: {time.time() - t0}')
