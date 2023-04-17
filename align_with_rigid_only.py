import copy
import numpy as np
import open3d as o3
import time
from probreg import cpd
import logging
import os
from scipy.spatial import KDTree


log = logging.getLogger('probreg')
log.setLevel(logging.DEBUG)
start_time = time.time()

# create a file to record the sequence of files read
file_list_path = 'file_list.txt'
with open(file_list_path, 'w') as f_list:
    # loop to read data files
    dir_path = r'path\to\segmentation\folder'
    filenames = []
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            filenames.append(path)
            f_list.write(path + '\n')
    x = len(filenames)

# load reference point cloud
pcd_file = r'path\to\segmentation'
reference = o3.io.read_point_cloud(pcd_file)
reference.remove_non_finite_points()

# Downsize to approximately 'num_points' of points using uniform downsample
num_points = 20000
every_k_points = 1000
downsampled_reference = reference.uniform_down_sample(every_k_points=every_k_points)
while len(downsampled_reference.points) < num_points:
    every_k_points -= 1
    downsampled_reference = reference.uniform_down_sample(every_k_points)
# downsize to a specific number of points
indices = np.random.choice(range(len(reference.points)), num_points)
downsampled_reference_array = np.asarray(reference.points)[indices]
downsampled_reference = o3.geometry.PointCloud()
downsampled_reference.points = o3.utility.Vector3dVector(downsampled_reference_array)

# Initialize array
aligned_instances = []
# Initialize KDTree
tree2 = KDTree(downsampled_reference_array)

# Loop for loading instances
for i in range(x):
    # Load instance
    every_k_points = 1000
    start_time = time.time()
    pcd_file = os.path.join(dir_path, filenames[i])
    instance = o3.io.read_point_cloud(pcd_file)
    instance.remove_non_finite_points()
    downsampled_instance = instance.uniform_down_sample(every_k_points=every_k_points)

    while len(downsampled_instance.points) < num_points:
        every_k_points -= 1
        downsampled_instance = instance.uniform_down_sample(every_k_points)

    indices = np.random.choice(range(len(instance.points)), num_points)
    downsampled_instance_array = np.asarray(instance.points)[indices]
    downsampled_instance = o3.geometry.PointCloud()
    downsampled_instance.points = o3.utility.Vector3dVector(downsampled_instance_array)

    # non-rigid CPD to spatially align
    tf_param, _, _ = cpd.registration_cpd(downsampled_instance, downsampled_reference, 'rigid')
    transformed_instance = copy.deepcopy(downsampled_instance)
    transformed_instance.points = tf_param.transform(transformed_instance.points)
    transformed_instance_array = np.array(transformed_instance.points)
    # create KDTree for instance
    tree1 = KDTree(transformed_instance_array)
    # use KDTree to find neighbouring corresponding points and rearrange instance points to correspond to reference
    dist, idx = tree1.query(downsampled_reference_array, k=1)
    correspondences = [(i, j) for i, j in enumerate(idx)]
    correspondences_array = np.array(correspondences)
    aligned_points= transformed_instance_array[correspondences_array[:, 1]]

    aligned_instances.append(aligned_points)

    print('elapsed_time=', time.time() - start_time)

# concatenate all aligned point clouds into a single array
aligned_instances_array = np.stack([np.asarray(pc) for pc in aligned_instances])
print('elapsed_time=', time.time() - start_time)
np.save("file_name", aligned_instances_array)





