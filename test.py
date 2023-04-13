import open3d as o3
import numpy as np
import vtk
from plane_fitting import fit_plane_to_points_scapula, fit_plane_to_points_glenoid, planes_version
import utils
import inclination_algorithms

# Read point cloud for scapula
pcd_file = r"C:\Users\xinglim\PycharmProjects\pythonProject\use_segmentations\p20190114_um01.ply"
scapula_pc = o3.io.read_point_cloud(pcd_file)
scapula_pc.remove_non_finite_points()
scapula_array = np.asarray(scapula_pc.points)

# Read point cloud for surface
pcd_file = r"C:\Users\xinglim\Documents\auto\p20190114_um01_surface.ply"
surface_pc = o3.io.read_point_cloud(pcd_file)
surface_pc.remove_non_finite_points()
surface_array = np.asarray(surface_pc.points)
# read point cloud into vtk format
surface_vtk = vtk.vtkPLYReader()
surface_vtk.SetFileName(pcd_file)
surface_vtk.Update()

# Read point cloud for slicer edge
pcd_file = r"C:\Users\xinglim\Documents\auto\p20190114_um01_edge.ply"
slicer_edge_pc = o3.io.read_point_cloud(pcd_file)
slicer_edge_pc.remove_non_finite_points()
slicer_edge_array = np.asarray(slicer_edge_pc.points)

# Fit sphere to glenoid cavity surface
sphere_radius, sphere_centre = utils.sphere_fit(surface_array)

# Find glenoid centre
glenoid_centre = utils.find_centroid(surface_array)

# Find inferior tip of scapula
inferior_tip = utils.find_inferior_tip(scapula_array)

# Find medial pole
medial_pole = utils.find_medial_pole(scapula_pc, inferior_tip, glenoid_centre)

# Automated glenoid edge extraction
auto_edge_array = utils.find_glenoid_edge_points(surface_vtk)

# Make scapula plane
scapula_plane = fit_plane_to_points_scapula(utils.stack_coordinates([glenoid_centre, medial_pole, inferior_tip]), True)

# Make glenoid planes
slicer_glenoid_plane = fit_plane_to_points_glenoid(slicer_edge_array, return_meta2=True)
auto_glenoid_plane = fit_plane_to_points_glenoid(auto_edge_array, return_meta2=True)

# Compute version
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
