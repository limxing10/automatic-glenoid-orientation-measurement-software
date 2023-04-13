import open3d as o3


def create_geometry_at_points(point, colour=[0,0,0], radius=1):
    """
    Function to create spheres at landmarks

    Args:
        point: numpy array of shape (3,), representing the coordinates of the point where the sphere will be created
        colour: numpy array of shape (3,), representing the color of the sphere in RGB format

    Returns:
        open3d.geometry.TriangleMesh: A `TriangleMesh` object containing the sphere geometry and color at the specified
                                      point coordinate
    """

    # Create an empty TriangleMesh object to store the geometries
    geometries = o3.geometry.TriangleMesh()
    # Create a sphere geometry with a specified radius of 2
    sphere = o3.geometry.TriangleMesh.create_sphere(radius=radius)
    # Translate the sphere to the specified point coordinates
    sphere.translate(point)
    # Add the translated sphere to the geometries
    geometries += sphere
    # Paint the geometries with the specified color
    geometries.paint_uniform_color(colour)
    # Return the geometries object with the sphere geometry and color
    return geometries
