import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

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
    # Create a figure with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=[f"{kernel1_name} Kernel", f"{kernel2_name} Kernel"])
    
    # If labels are provided, order the kernel matrices accordingly
    if y is not None:
        # Get indices for ordering
        indices = np.argsort(y)
        K1_ordered = K1[indices][:, indices]
        K2_ordered = K2[indices][:, indices]
    else:
        K1_ordered = K1
        K2_ordered = K2
    
    # Add heatmap for first kernel
    fig.add_trace(
        go.Heatmap(
            z=K1_ordered,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Similarity", x=0.46)
        ),
        row=1, col=1
    )
    
    # Add heatmap for second kernel
    fig.add_trace(
        go.Heatmap(
            z=K2_ordered,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Similarity", x=1.02)
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        width=900,
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
    # Create a figure with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=["Original Data", f"Data Transformed with {kernel_type} Kernel"])
    
    # Ensure we only use first two dimensions for visualization
    X_vis = X if X.shape[1] <= 2 else X[:, :2]
    
    # For higher dimensional data, create a note about dimensionality reduction
    dim_note = "" if X.shape[1] <= 2 else f"(showing first 2 of {X.shape[1]} dimensions)"
    
    # Create dataframes for plotting
    df_original = pd.DataFrame({
        'x': X_vis[:, 0],
        'y': X_vis[:, 1] if X_vis.shape[1] > 1 else np.zeros(X_vis.shape[0]),
        'class': y
    })
    
    df_transformed = pd.DataFrame({
        'x': X_transformed[:, 0],
        'y': X_transformed[:, 1],
        'class': y
    })
    
    # Add scatter plot for original data
    fig.add_trace(
        go.Scatter(
            x=df_original['x'],
            y=df_original['y'],
            mode='markers',
            marker=dict(
                size=10,
                color=df_original['class'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Class", x=0.46)
            ),
            text=df_original['class'],
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add scatter plot for transformed data
    fig.add_trace(
        go.Scatter(
            x=df_transformed['x'],
            y=df_transformed['y'],
            mode='markers',
            marker=dict(
                size=10,
                color=df_transformed['class'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Class", x=1.02)
            ),
            text=df_transformed['class'],
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        width=900,
        title_text=f"Data Transformation with {kernel_type} Kernel ({data_type} Data {dim_note})",
    )
    
    # Update axes
    fig.update_xaxes(title_text="Feature 1", row=1, col=1)
    fig.update_yaxes(title_text="Feature 2", row=1, col=1)
    fig.update_xaxes(title_text="Component 1", row=1, col=2)
    fig.update_yaxes(title_text="Component 2", row=1, col=2)
    
    return fig

def plot_eigenspectrum(K, kernel_type, title=None):
    """
    Plot eigenvalues of a kernel matrix to show the indefiniteness
    
    Parameters:
    K: numpy array, kernel matrix
    kernel_type: str, type of kernel
    title: str, optional title for the plot
    
    Returns:
    fig: plotly figure
    """
    # Compute eigenvalues
    eigvals = np.linalg.eigvalsh(K)
    
    # Create indices for plotting
    indices = np.arange(len(eigvals))
    
    # Create color array: positive eigenvalues in blue, negative in red
    colors = ['blue' if val >= 0 else 'red' for val in eigvals]
    
    # Sort eigenvalues by magnitude (absolute value) for better visualization
    sorted_idx = np.argsort(np.abs(eigvals))[::-1]
    eigvals_sorted = eigvals[sorted_idx]
    colors_sorted = [colors[i] for i in sorted_idx]
    
    # Create figure
    if title is None:
        title = f"Eigenspectrum of {kernel_type} Kernel Matrix"
    
    fig = go.Figure()
    
    # Add bar plot for eigenvalues
    fig.add_trace(
        go.Bar(
            x=np.arange(len(eigvals_sorted)),
            y=eigvals_sorted,
            marker_color=colors_sorted,
            name='Eigenvalues'
        )
    )
    
    # Add a line at zero for reference
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=len(eigvals_sorted) - 1,
        y1=0,
        line=dict(color="black", width=2, dash="dash"),
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Eigenvalue Index (sorted by magnitude)",
        yaxis_title="Eigenvalue",
        height=400,
        width=700,
        showlegend=False,
        annotations=[
            dict(
                x=0.05,
                y=0.95,
                xref="paper",
                yref="paper",
                text="Positive definite component",
                showarrow=False,
                font=dict(color="blue", size=12)
            ),
            dict(
                x=0.05,
                y=0.85,
                xref="paper",
                yref="paper",
                text="Negative definite component",
                showarrow=False,
                font=dict(color="red", size=12)
            )
        ]
    )
    
    return fig

def plot_synergy_matrix(drug_pairs_df):
    """
    Plot a heat map of synergy scores between drug pairs
    
    Parameters:
    drug_pairs_df: pandas DataFrame with columns Drug_A, Drug_B, and Synergy_Score
    
    Returns:
    fig: plotly figure
    """
    # Get unique drugs
    drugs = sorted(list(set(drug_pairs_df['Drug_A'].unique()) | set(drug_pairs_df['Drug_B'].unique())))
    
    # Create empty matrix
    n_drugs = len(drugs)
    synergy_matrix = np.zeros((n_drugs, n_drugs))
    
    # Create drug to index mapping
    drug_to_idx = {drug: i for i, drug in enumerate(drugs)}
    
    # Fill the matrix
    for _, row in drug_pairs_df.iterrows():
        i = drug_to_idx[row['Drug_A']]
        j = drug_to_idx[row['Drug_B']]
        synergy_matrix[i, j] = row['Synergy_Score']
        synergy_matrix[j, i] = row['Synergy_Score']  # Make it symmetric
    
    # Create figure
    fig = go.Figure()
    
    # Add heatmap
    fig.add_trace(
        go.Heatmap(
            z=synergy_matrix,
            x=drugs,
            y=drugs,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Synergy Score"),
            text=[[f"{drugs[i]} + {drugs[j]}: {synergy_matrix[i, j]:.2f}" 
                   for j in range(n_drugs)] for i in range(n_drugs)],
            hoverinfo="text"
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Drug Synergy Matrix",
        xaxis_title="Drug B",
        yaxis_title="Drug A",
        height=600,
        width=700
    )
    
    return fig