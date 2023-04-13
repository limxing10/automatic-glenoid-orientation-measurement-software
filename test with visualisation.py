# Import necessary libraries
import open3d as o3
import numpy as np
import vtk
from plane_fitting import fit_plane_to_points_scapula, fit_plane_to_points_glenoid, planes_version
import utils
import inclination_algorithms
import visualisation_utils

# Read point cloud for scapula
# Load and preprocess point cloud data for scapula
pcd_file = r"C:\Users\xinglim\PycharmProjects\pythonProject\use_segmentations\p20190114_um01.ply"
scapula_pc = o3.io.read_point_cloud(pcd_file)
scapula_pc.remove_non_finite_points()
scapula_array = np.asarray(scapula_pc.points)

# Read point cloud for surface
# Load and preprocess point cloud data for surface
pcd_file = r"C:\Users\xinglim\Documents\auto\p20190114_um01_surface.ply"
surface_pc = o3.io.read_point_cloud(pcd_file)
surface_pc.remove_non_finite_points()
surface_array = np.asarray(surface_pc.points)

# Read point cloud into vtk format
surface_vtk = vtk.vtkPLYReader()
surface_vtk.SetFileName(pcd_file)
surface_vtk.Update()

# Read point cloud for slicer edge
# Load and preprocess point cloud data for slicer edge
pcd_file = r"C:\Users\xinglim\Documents\auto\p20190114_um01_edge.ply"
slicer_edge_pc = o3.io.read_point_cloud(pcd_file)
slicer_edge_pc.remove_non_finite_points()
slicer_edge_array = np.asarray(slicer_edge_pc.points)

# Fit sphere to glenoid cavity surface
# Fit a sphere to the glenoid cavity surface
sphere_radius, sphere_centre = utils.sphere_fit(surface_array)

# Find glenoid centre
# Find the centroid of the glenoid cavity surface
glenoid_centre = utils.find_centroid(surface_array)

# Find inferior tip of scapula
# Find the inferior tip of the scapula
inferior_tip = utils.find_inferior_tip(scapula_array)

# Find medial pole
# Find the medial pole of the scapula
medial_pole = utils.find_medial_pole(scapula_pc, inferior_tip, glenoid_centre)

# Automated glenoid edge extraction
# Extract the glenoid edge points using automated methods
auto_edge_array = utils.find_glenoid_edge_points(surface_vtk)

# Make scapula plane
# Fit a plane to the scapula using glenoid centre, medial pole, and inferior tip points
scapula_plane = fit_plane_to_points_scapula(utils.stack_coordinates([glenoid_centre, medial_pole, inferior_tip]), True)

# Make glenoid planes
# Fit planes to the slicer edge and automated edge points
slicer_glenoid_plane = fit_plane_to_points_glenoid(slicer_edge_array, return_meta2=True)
auto_glenoid_plane = fit_plane_to_points_glenoid(auto_edge_array, return_meta2=True)

# Compute version
# Compute the version (angle) between scapula plane and glenoid planes
slicer_version = planes_version(scapula_plane[2], slicer_glenoid_plane[2])
auto_version = planes_version(scapula_plane[2], auto_glenoid_plane[2])
# Create glenoid normal
glenoid_normal = utils.create_line_from_points([glenoid_centre, sphere_centre])

# Create medial-pole-inferior-tip-axis
med_pole_inf_tip = utils.create_line_from_points([medial_pole, inferior_tip])

# Compute inclination
transverse_inclination = inclination_algorithms.transverse_axis_inclination(glenoid_centre, medial_pole, glenoid_normal, scapula_plane)
med_pole_inf_tip_inclination = inclination_algorithms.medpole_inf_tip_inclination(sphere_centre, glenoid_centre, med_pole_inf_tip, scapula_plane)

# Print results
print("slicer_version: ", slicer_version)
print("auto_version: ", auto_version)
print('transverse_inclination:', transverse_inclination)
print('med_pole_inf_tip_inclination:', med_pole_inf_tip_inclination)

# Visuals
# Plot the sphere and the point cloud
fig = o3.visualization.Visualizer()
fig.create_window()

# Create sphere mesh
mesh_sphere = o3.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
mesh_sphere.translate(sphere_centre)

# Create point cloud for glenoid center
glenoid_centre_pc = visualisation_utils.create_geometry_at_points(glenoid_centre)

# Create transverse axis line
transverse_axis = utils.create_line_from_points([medial_pole, glenoid_centre])

# Set colors for visual elements
mesh_sphere.paint_uniform_color([0.572, 0.725, 0.784])
glenoid_centre_pc.paint_uniform_color([0, 0, 0])
glenoid_normal.paint_uniform_color([0, 0, 0])
transverse_axis.paint_uniform_color([0, 0, 0])

# Add visual elements to the figure
fig.add_geometry(mesh_sphere)
scapula_pc.paint_uniform_color([0, 0, 0])
fig.add_geometry(scapula_pc)
fig.add_geometry(glenoid_centre_pc)
fig.add_geometry(glenoid_normal)
fig.add_geometry(transverse_axis)

# Create point clouds for inferior tip and medial pole
inferior_tip_pc = visualisation_utils.create_geometry_at_points(inferior_tip)
medial_pole_pc = visualisation_utils.create_geometry_at_points(medial_pole)

# Set colors for inferior tip and medial pole point clouds
inferior_tip_pc.paint_uniform_color([0, 1, 0])
medial_pole_pc.paint_uniform_color([0, 1, 0])

# Add inferior tip and medial pole point clouds to the figure
fig.add_geometry(inferior_tip_pc)
fig.add_geometry(medial_pole_pc)

# Set color for surface point cloud
surface_pc.paint_uniform_color([1, 0, 0])

# Add surface point cloud to the figure
fig.add_geometry(surface_pc)

# Set color for slicer edge point cloud
slicer_edge_pc.paint_uniform_color([0, 1, 0])

# Add slicer edge point cloud to the figure
fig.add_geometry(slicer_edge_pc)

# Set color for medial-pole-inferior-tip-axis
med_pole_inf_tip.paint_uniform_color([0, 1, 0])

# Add medial-pole-inferior-tip-axis to the figure
fig.add_geometry(med_pole_inf_tip)

# Run the visualizer
fig.run()