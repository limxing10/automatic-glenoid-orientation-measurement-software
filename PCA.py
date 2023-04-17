import pyssam
import vtk
import pyvista as pv
import pyacvd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# initialise pyssam
landmark_coordinates = np.load('final_run_array.npy')
ssm_obj = pyssam.SSM(landmark_coordinates)
ssm_obj.create_pca_model(ssm_obj.landmarks_columns_scale)
mean_shape_columnvector = ssm_obj.compute_dataset_mean()
mean_shape = mean_shape_columnvector.reshape(-1, 3)
shape_model_components = ssm_obj.pca_model_components


# Define some plotting functions
def plot_cumulative_variance_plt(explained_variance, target_variance=-1):
    """
    Function to plot cumulative variance explained by principal components.

    Parameters:
        - explained_variance (numpy array): Array of explained variance for each principal component
        - target_variance (float): Target variance (default: -1)

    Returns:
        None
    """

    # Create an array of number of components
    number_of_components = np.arange(0, len(explained_variance))+1

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(1, 1)

    # Set marker style and color for the plot
    color = "blue"
    ax.plot(number_of_components, explained_variance*100.0, marker="o", ms=2, color=color, mec=color, mfc=color)

    # If a target variance is specified, plot a horizontal line at that target variance
    if target_variance > 0.0:
        ax.axhline(target_variance*100.0)

    # Set labels and title for the plot
    ax.set_ylabel("Variance [%]")
    ax.set_xlabel("Number of components")
    ax.set_title("Cumulative Variance Explained by Principal Components for 20000 point dataset")

    # Add grid to x-axis
    ax.grid(axis="x")

    # Display the plot
    plt.show()


def plot_cumulative_variance_seaborn(explained_variance, target_variance=-1):
    """
    Function to plot cumulative variance explained by principal components using Seaborn library.

    Parameters:
        - explained_variance (numpy array): Array of explained variance for each principal component
        - target_variance (float): Target variance (default: -1)

    """

    # Create an array of number of components
    number_of_components = np.arange(0, len(explained_variance))+1

    # Set Seaborn style and palette for the plot
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(1, 1)

    # Set marker style and color for the plot
    color = sns.color_palette()[0]
    ax.plot(number_of_components, explained_variance*100.0, marker="o", ms=2, color=color, mec=color, mfc=color)

    # If a target variance is specified, plot a horizontal line at that target variance with orange color
    if target_variance > 0.0:
        ax.axhline(target_variance*100.0, color= sns.color_palette()[1])

    # Set labels and title for the plot
    ax.set_ylabel("Variance [%]")
    ax.set_xlabel("Number of Components")
    ax.set_title("Cumulative Variance Explained by Principal Components for 2000 point dataset")

    # Set y-axis limit to start from 0
    ax.set_ylim(ymin=0)

    # Display the plot
    plt.show()


def plot_shape_mode_vtk(
    mean_shape_columnvector,
    original_shape_parameter_vector,
    shape_model_components,
    mode_to_plot=0,
    std_dev=0,
    clusters=50000
):
    """
    Function to plot a shape mode using VTK (Visualization Toolkit).

    Parameters:
        - mean_shape_columnvector (numpy array): Mean shape as a column vector
        - mean_shape (numpy array): Mean shape as a numpy array
        - original_shape_parameter_vector (numpy array): Original shape parameter vector
        - shape_model_components (numpy array): Shape model components
        - mode_to_plot (int): Mode to plot (default: 0)
        - std_dev (float): Standard deviation for selected mode (default: 0)
        - clusters (int): Number of clusters for mesh simplification (default: 50000)

    """
    # Create vtk points and cells objects for the selected mode
    shape_parameter_vector = original_shape_parameter_vector.copy()
    shape_parameter_vector[mode_to_plot] = std_dev
    mode_coords = ssm_obj.morph_model(
        mean_shape_columnvector,
        shape_model_components,
        shape_parameter_vector
    ).reshape(-1, 3)

    points = vtk.vtkPoints()
    for i, (x, y, z) in enumerate(mode_coords):
        points.InsertPoint(i, x, y, z)

    cells = vtk.vtkCellArray()
    for i in range(mode_coords.shape[0]):
        cells.InsertNextCell(1)
        cells.InsertCellPoint(i)

    # Create vtk polydata object for the selected mode
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetVerts(cells)

    delaunay = vtk.vtkDelaunay3D()
    delaunay.SetInputData(polydata)
    delaunay.SetTolerance(0.01)
    delaunay.SetAlpha(0.1)
    delaunay.Update()
    outputMesh = delaunay.GetOutput()

    geometryFilter = vtk.vtkGeometryFilter()
    geometryFilter.SetInputData(outputMesh)
    geometryFilter.Update()
    polyData = geometryFilter.GetOutput()

    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputData(polyData)
    smoother.SetNumberOfIterations(50)
    smoother.Update()
    polyData = smoother.GetOutput()

    inputMesh = pv.wrap(polyData)
    clus = pyacvd.Clustering(inputMesh)
    clus.subdivide(3)
    clus.cluster(clusters)

    outputMesh = clus.create_mesh()

    # Update output file name based on input parameters
    output_file_name = f"mode{mode_to_plot+1}_sd{std_dev}.ply"
    outputMesh.save(output_file_name)
    plotter = pv.Plotter()
    plotter.add_mesh(outputMesh)
    plotter.show()


mode = 0
stand_dev = np.sqrt(ssm_obj.pca_object.explained_variance_ratio_[mode])
stans_dev_to_plot = 5*stand_dev

print(f"explained variance is {ssm_obj.pca_object.explained_variance_ratio_[mode]}")
plot_cumulative_variance_seaborn(np.cumsum(ssm_obj.pca_object.explained_variance_ratio_), 0.9)

# plot
plot_shape_mode_vtk(
    mean_shape_columnvector,
    ssm_obj.model_parameters,
    ssm_obj.pca_model_components,
    mode_to_plot=mode,
    std_dev=stans_dev_to_plot
)
