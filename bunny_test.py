import numpy as np
import open3d as o3d
import FPS

bunny = o3d.data.BunnyMesh()
mesh = o3d.io.read_triangle_mesh(bunny.path)

pcd = mesh.sample_points_uniformly(10000)
o3d.visualization.draw_geometries([pcd])

points = np.asarray(pcd.points)
inds = FPS.farthest_point_sampling(points, 2048, np.random.randint(0,10000))
points = points[inds, :]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
o3d.visualization.draw_geometries([pcd])
