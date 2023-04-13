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

# Create a file to record the sequence of files read
file_list_path = 'file_list.txt'
with open(file_list_path, 'w') as f_list:
    # Loop to read data files
    dir_path = r'\path\to\segmentations_folder'
    filenames = []
    for path in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, path)):
            filenames.append(path)
            f_list.write(path + '\n')
    x = len(filenames)

# Load reference point cloud
pcd_file = r'path\to\segmentation'
reference= o3.io.read_point_cloud(pcd_file)
reference.remove_non_finite_points()

num_points = 2000

# Downsize to approximately 'num_points' of points using uniform downsample
every_k_points = 1000
downsampled_reference = reference.uniform_down_sample(every_k_points=every_k_points)
# Downsize to a specific number of points
while len(downsampled_reference.points) < num_points:
    every_k_points -= 1
    downsampled_reference = reference.uniform_down_sample(every_k_points)
indices = np.random.choice(range(len(reference.points)), num_points)
downsampled_reference_array = np.asarray(reference.points)[indices]
downsampled_reference = o3.geometry.PointCloud()
downsampled_reference.points = o3.utility.Vector3dVector(downsampled_reference_array)

# Initialize array
aligned_instances = []
# Initialize KDTree
tree2 = KDTree(downsampled_reference_array)

# Empirically determined non-rigid registration parameters
tols = [1e-3, 1e-6, 1e-9]
lmds = [2, 2, 2]

# Loop for loading instances
for i in range(x):
    # Load instance
    every_k_points = 1000
    start_time = time.time()
    filename = os.path.join(dir_path, filenames[i])
    instance = o3.io.read_point_cloud(filename)

    # Downsample instances
    instance.remove_non_finite_points()
    downsampled_instance = instance.uniform_down_sample(every_k_points=every_k_points)
    while len(downsampled_instance.points) < num_points:
        every_k_points -= 1
        downsampled_instance = instance.uniform_down_sample(every_k_points)
    indices = np.random.choice(range(len(instance.points)), num_points)
    downsampled_instance_array = np.asarray(instance.points)[indices]
    downsampled_instance = o3.geometry.PointCloud()
    downsampled_instance.points = o3.utility.Vector3dVector(downsampled_instance_array)

    # Initialize correspondence transference by first rigid CPD registration for more efficient non-rigid later
    tf_param, _, _ = cpd.registration_cpd(downsampled_reference, downsampled_instance, 'rigid')
    transformed_instance = copy.deepcopy(downsampled_instance)
    transformed_instance.points = tf_param.transform(transformed_instance.points)
    transformed_instance_array = np.array(transformed_instance.points)

    # First iteration of non-rigid CPD
    tf_param, _, _ = cpd.registration_cpd(downsampled_reference, downsampled_instance, tf_type_name='nonrigid', lmd=1e-5,
                                          tol=1)
    result = copy.deepcopy(downsampled_reference)
    result.points = tf_param.transform(result.points)
    result_array = np.array(result.points)

    # loop for next three iterations of non-rigid CPD
    for i in range(3):
        # calculate average between instance and interim
        data = np.stack([result_array, downsampled_instance_array], axis=0)
        result_avg = np.mean(data, axis=0)
        result_avg_pcd = o3.geometry.PointCloud()
        result_avg_pcd.points = o3.utility.Vector3dVector(result_avg)

        # perform following non-rigid CPD using the average as the source
        tf_param, _, _ = cpd.registration_cpd(result_avg_pcd, downsampled_instance, tf_type_name='nonrigid',
                                              lmd=lmds[i], tol=tols[i])

        # update result with the transformed points
        result.points = tf_param.transform(result.points)
        result_array = np.array(result.points)

    # non-rigid CPD to align the now with correspondence transference instances to spatially align with reference and
    # hence each other
    tf_param, _, _ = cpd.registration_cpd(result, downsampled_reference, 'rigid')
    transformed_instance = copy.deepcopy(result)
    transformed_instance.points = tf_param.transform(transformed_instance.points)
    transformed_instance_array = np.array(transformed_instance.points)

    # create KDTree for instance
    tree1 = KDTree(transformed_instance_array)

    # use KDTree to find neighbouring corresponding points
    dist, idx = tree1.query(downsampled_reference_array, k=1)

    correspondences = [(i, j) for i, j in enumerate(idx)]
    correspondences_array = np.array(correspondences)
    aligned_points = transformed_instance_array[correspondences_array[:, 1]]

    aligned_instances.append(aligned_points)

# concatenate all aligned point clouds into a single array
aligned_instances_array = np.stack([np.asarray(pc) for pc in aligned_instances])

print('elapsed_time=', time.time() - start_time)
np.save("file_name", aligned_instances_array)





