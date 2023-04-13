import numpy as np
import math
import random
import vtk
import open3d as o3


# Extract coordinates from point cloud
def extract_coordinates_from_point_cloud(point_cloud):
    """
    Extracts X, Y, Z coordinates from a point cloud.

    Args:
        point_cloud (numpy.ndarray): Input point cloud in the shape of (n_points, 3).

    Returns:
        X (numpy.ndarray): X coordinates as a 1D numpy array.
        Y (numpy.ndarray): Y coordinates as a 1D numpy array.
        Z (numpy.ndarray): Z coordinates as a 1D numpy array.
    """
    # Convert point cloud to numpy array
    point_cloud_array = np.asarray(point_cloud)

    # Extract X, Y, Z coordinates
    X = point_cloud_array[:, 0]
    Y = point_cloud_array[:, 1]
    Z = point_cloud_array[:, 2]

    return X, Y, Z


# Fit a best-fit sphere to the point cloud data
def sphere_fit(surface):
    """
    Fits a sphere using the least-squares method to the given point cloud data.

    Args:
        surface (numpy.ndarray): Input point cloud in the shape of (n_points, 3).

    Returns:
        tuple: A tuple containing the radius of the sphere, and the X, Y, and Z coordinates of the centre of the sphere.
    """

    [sX, sY, sZ] = extract_coordinates_from_point_cloud(surface)
    # Assemble the A matrix
    spx = np.array(sX)
    spy = np.array(sY)
    spz = np.array(sZ)
    A = np.zeros((len(spx), 4))
    A[:, 0] = spx * 2   # X coordinates multiplied by 2
    A[:, 1] = spy * 2   # Y coordinates multiplied by 2
    A[:, 2] = spz * 2   # Z coordinates multiplied by 2
    A[:, 3] = 1         # Column of ones for bias term

    # Assemble the f matrix
    f = np.zeros((len(spx), 1))
    f[:, 0] = spx * spx + spy * spy + spz * spz  # Sum of squares of X, Y, Z coordinates

    # Solve the linear system using least squares
    C, residuals, rank, singular_values = np.linalg.lstsq(A, f, rcond=-1)

    # Solve for the radius
    t = C[0] * C[0] + C[1] * C[1] + C[2] * C[2] + C[3]
    radius = np.sqrt(t)
    x = C[0]
    y = C[1]
    z = C[2]
    sphere_centre = [x,y,z]

    return radius, sphere_centre


# Given a set of 3D points, returns the centroid as a numpy array.
def find_centroid(points):
    """
    Given a set of 3D points, returns the centroid as a numpy array.

    Args:
        points (ndarray): An array of 3D points represented as a numpy array.

    Returns:
        ndarray: A numpy array representing the centroid of the points.
    """
    num_points = len(points)
    centroid = np.sum(points, axis=0) / num_points
    return centroid


# Computes the Euclidean distance between two points.
def distance(a, b):
    """
    Computes the Euclidean distance between two points.

    Args:
        a (tuple): A tuple representing the coordinates of the first point.
        b (tuple): A tuple representing the coordinates of the second point.

    Returns:
        float: The Euclidean distance between the two points.
    """
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


#  Computes the farthest pair of points in a set of points.
def farthest_pair(points, max_iterations=20, convergence_threshold=1e-3):
    """
    Computes the farthest pair of points in a set of points.

    Args:
        points (list): A list of points represented as tuples or other comparable objects.
        max_iterations (int): Maximum number of iterations for convergence.
        convergence_threshold (float): Threshold for convergence.

    Returns:
        tuple: A tuple containing the two farthest points (q, r).
    """
    # Pick a random point as the starting point
    p = random.choice(points)

    # Find the farthest point from the starting point
    q = max(points, key=lambda x: distance(p, x))

    # Find the farthest point from q
    r = max(points, key=lambda x: distance(q, x))

    # Convergence criteria: Maximum number of iterations and change in distance below threshold
    iterations = 0
    prev_distance = distance(q, r)
    while iterations < max_iterations:
        # Compute the midpoint of the farthest pair
        mid = tuple((a + b) / 2 for a, b in zip(q, r))

        # Find the farthest point from the midpoint
        s = max(points, key=lambda x: distance(mid, x))

        # If s is closer to q than r, update r
        if distance(s, q) > distance(s, r):
            r = s
        # If s is closer to r than q, update q
        elif distance(s, r) > distance(s, q):
            q = s
        # If s is equidistant from q and r, we've converged
        else:
            break

        # Check for convergence based on change in distance
        current_distance = distance(q, r)
        if abs(current_distance - prev_distance) < convergence_threshold:
            break

        prev_distance = current_distance
        iterations += 1

    return q, r


# Finds the inferior tip of a set of points by computing the farthest pair of points along the Z-axis.
def find_inferior_tip(points):
    """
    Finds the inferior tip of a set of points by computing the farthest pair of points along the Z-axis.

    Args:
        points (list): A list of points represented as tuples or other comparable objects.

    Returns:
        tuple: A tuple containing the coordinates of the inferior tip of the points.
    """
    # Compute the farthest pair of points
    q, p = farthest_pair(points)

    # Ensure that the lower point is selected as the inferior point
    if q[2] < p[2]:
        inferior = q
    else:
        inferior = p

    return inferior


# Project points onto a plane defined by its centre and normal.
def project_points_onto_plane(points, centre, normal):
    """
    Project points onto a plane defined by its centre and normal.

    Args:
        points (np.ndarray): A numpy array of size n by 3 representing the points.
        centre (np.ndarray): A numpy array of size 3 representing the centre of the plane.
        normal (np.ndarray): A numpy array of size 3 representing the normal vector of the plane.

    Returns:
        np.ndarray: A numpy array of size n by 3 representing the projected points.
    """
    # Calculate the distances from each point to the plane
    distances = np.dot(points - centre, normal)
    # Project the points onto the plane using the distances and normal vector
    projected_points = points - np.outer(distances, normal)
    # Return the projected points
    return projected_points


def make_plane_model(centre, normal):
    """
    Creates a plane model using the specified centre and normal vector.

    Args:
        centre (np.ndarray): A numpy array of size 3 representing the centre of the plane.
        normal (np.ndarray): A numpy array of size 3 representing the normal vector of the plane.

    Returns:
        vtk.vtkPlaneSource: A vtkPlaneSource object representing the plane model.
    """
    plane = vtk.vtkPlaneSource()
    plane.Setcentre(centre)
    plane.SetNormal(normal)
    plane.Update()

    return plane


# to find glenoid edge points
def find_glenoid_edge_points(reader):
    """
    Find glenoid edge points from a VTK polydata reader.

    Args:
        reader (vtk.vtkPolyDataReader): VTK polydata reader.

    Returns:
        np.ndarray: Numpy array containing the glenoid edge points.
    """
    boundaryEdges = vtk.vtkFeatureEdges()
    boundaryEdges.SetInputConnection(reader.GetOutputPort())
    boundaryEdges.SetBoundaryEdges(True)
    boundaryEdges.Update()
    outputModel = vtk.vtkPolyData()
    outputModel.ShallowCopy(boundaryEdges.GetOutput())

    connect = vtk.vtkPolyDataConnectivityFilter()
    connect.SetInputData(outputModel)
    connect.SetExtractionModeToLargestRegion()
    connect.Update()
    largest = vtk.vtkPolyData()
    largest.ShallowCopy(connect.GetOutput())

    points = largest.GetPoints()
    num_points = points.GetNumberOfPoints()

    points_array = np.zeros((num_points, 3))
    for i in range(num_points):
        points_array[i, :] = points.GetPoint(i)

    return points_array


# create line connecting two points
def create_line_from_points(points):
    """
    Creates a LineSet from a list of points.

    Args:
        points (list): List of points as [point1, point2].

    Returns:
        open3d.geometry.LineSet: LineSet object representing the line segment between the two points.
    """
    lines = o3.utility.Vector2iVector([[0, 1]])
    points = o3.utility.Vector3dVector(points)
    lineset = o3.geometry.LineSet(points=points, lines=lines)
    return lineset


# find medial pole
def find_medial_pole(scapula, inferior_tip, glenoid_centre):
    """
    Finds the medial pole of the scapula based on two baseline points.

    Args:
        scapula (open3d.geometry.PointCloud): Input point cloud of the scapula.
        inferior_tip (numpy.ndarray): Baseline point 1 as a 1D numpy array of shape (3,).
        glenoid_centre (numpy.ndarray): Baseline point 2 as a 1D numpy array of shape (3,).

    Returns:
        med_bord (numpy.ndarray): Coordinates of the farthest point from the baseline points
                                  as a 1D numpy array of shape (3,).
    """
    # Create a new point cloud from the two baseline points
    baseline_cloud = o3.geometry.PointCloud()
    baseline_cloud.points = o3.utility.Vector3dVector(np.vstack((inferior_tip, glenoid_centre)))

    # Compute the distances between each point in the scapula point cloud and the baseline points
    distances = scapula.compute_point_cloud_distance(baseline_cloud)

    # Get the farthest point from the baseline points
    farthest_point_index = np.argmax(distances)

    # Extract the coordinates of the farthest point
    scapula_array = np.asarray(scapula.points)
    med_pole = np.asarray(scapula_array[farthest_point_index])

    return med_pole


def stack_coordinates(coordinates):
    """
    Stack coordinates in a vertical orientation.

    Args:
        coordinates (numpy.ndarray): list of coordinates to be stacked vertically

    Returns:
        stacked_coordinates (numpy.ndarray): A 3x3 NumPy array with the stacked coordinates.
    """
    stacked_coordinates = np.vstack([coordinates])
    return stacked_coordinates
