import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def plot_kernel_matrix(K1, K2, kernel1_name, kernel2_name, y=None):
    """
    Plot two kernel matrices side by side for comparison
    
    Parameters:
    K1: numpy array, first kernel matrix
    K2: numpy array, second kernel matrix
    kernel1_name: str, name of the first kernel
    kernel2_name: str, name of the second kernel
    y: numpy array, optional labels for ordering the matrices
    
    Returns:
    fig: plotly figure
    """
    # Create a subplot with 1 row and 2 columns
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=(f"{kernel1_name} Kernel", f"{kernel2_name} Kernel")
    )
    
    # If labels are provided, reorder the matrices by class
    if y is not None and len(np.unique(y)) < len(y) / 2:  # Only reorder if we have meaningful classes
        # Get sort indices by class
        sort_idx = np.argsort(y)
        K1_sorted = K1[sort_idx][:, sort_idx]
        K2_sorted = K2[sort_idx][:, sort_idx]
    else:
        K1_sorted = K1
        K2_sorted = K2
    
    # Add heatmaps
    fig.add_trace(
        go.Heatmap(
            z=K1_sorted,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Similarity",
                x=0.46
            )
        ),
        row=1, col=1
    )
    
    # For Kreĭn-space kernel, use a diverging colorscale to show negative values
    if kernel2_name == "Kreĭn-Space":
        colorscale = 'RdBu'
        # Center the colorscale at zero
        zmin = min(0, K2_sorted.min())
        zmax = max(0, K2_sorted.max())
        absmax = max(abs(zmin), abs(zmax))
        zmin, zmax = -absmax, absmax
    else:
        colorscale = 'Viridis'
        zmin, zmax = None, None
    
    fig.add_trace(
        go.Heatmap(
            z=K2_sorted,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            showscale=True,
            colorbar=dict(
                title="Similarity",
                x=1.02
            )
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        width=800,
        title_text="Kernel Matrix Comparison",
    )
    
    return fig


def plot_transformed_data(X, X_transformed, y, kernel_type, data_type):
    """
    Plot original data and kernel-transformed data side by side
    
    Parameters:
    X: numpy array, original data
    X_transformed: numpy array, transformed data
    y: numpy array, labels
    kernel_type: str, type of kernel used
    data_type: str, type of data generated
    
    Returns:
    fig: plotly figure
    """
    # Create a subplot with 1 row and 2 columns
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=("Original Data", f"After {kernel_type} Kernel Transformation")
    )
    
    # Determine if we should plot in 2D or 3D
    if X.shape[1] > 2:
        # Use only first two dimensions for visualization
        X_plot = X[:, :2]
    else:
        X_plot = X
    
    # Convert labels to strings for consistent plotting
    y_str = [str(label) for label in y]
    
    # Add scatter plot for original data
    fig.add_trace(
        go.Scatter(
            x=X_plot[:, 0],
            y=X_plot[:, 1] if X_plot.shape[1] > 1 else np.zeros(X_plot.shape[0]),
            mode='markers',
            marker=dict(
                size=8,
                color=y,
                colorscale='Viridis',
                showscale=False
            ),
            text=y_str,
            name='Original'
        ),
        row=1, col=1
    )
    
    # Add scatter plot for transformed data
    fig.add_trace(
        go.Scatter(
            x=X_transformed[:, 0],
            y=X_transformed[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=y,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Class",
                    x=1.02
                )
            ),
            text=y_str,
            name='Transformed'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        width=800,
        title_text=f"Effect of {kernel_type} Kernel on {data_type} Data",
    )
    
    # Update axes
    fig.update_xaxes(title_text="Dimension 1", row=1, col=1)
    fig.update_yaxes(title_text="Dimension 2", row=1, col=1)
    fig.update_xaxes(title_text="Transformed Dimension 1", row=1, col=2)
    fig.update_yaxes(title_text="Transformed Dimension 2", row=1, col=2)
    
    return fig
