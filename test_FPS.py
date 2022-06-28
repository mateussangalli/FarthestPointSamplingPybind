import FPS
import numpy as np
import time

print('5 points') 
t0 = time.time()
a = np.random.rand(5, 3)
print(a)
dist_matrix = np.sum((a[:,np.newaxis,:] - a[np.newaxis,:,:])**2, 2)
print(dist_matrix)

b = FPS.farthest_point_sampling(a, 3, np.random.randint(0,4))

print(f'output_shape: {b.shape}')
print('indices:')
print(b)
print(f'elapsed time: {time.time() - t0}')


print('10000 points') 
t0 = time.time()
a = np.random.rand(10000, 3)
dist_matrix1 = np.sum((a[:,np.newaxis,:] - a[np.newaxis,:,:])**2, 2)

b = FPS.farthest_point_sampling(a, 2048, 42)
b2 = FPS.farthest_point_sampling(a, 10, 42)
print(f'perm1: {b[:10]}')
print(f'perm2: {b2}')
a2 = a[b, :]
dist_matrix2 = np.sum((a2[:,np.newaxis,:] - a2[np.newaxis,:,:])**2, 2)
print(f'output_shape: {b.shape}')
print(f'elapsed time: {time.time() - t0}')
print(f'original distance matrix mean: {dist_matrix1.mean()}')
print(f'new distance matrix mean: {dist_matrix2.mean()}')


print('10 x 10000 points') 
t0 = time.time()
for i in range(10):
    a = np.random.rand(10000, 3)
    b = FPS.farthest_point_sampling(a, 2048, np.random.randint(0,10000))
print(f'elapsed time: {time.time() - t0}')
