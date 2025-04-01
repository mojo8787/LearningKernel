import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.manifold import TSNE
import io

from kernel_utils import compute_kernel_matrix, compute_krein_kernel_matrix, apply_kernel_transformation
from visualization import plot_kernel_matrix, plot_transformed_data
from data_generator import generate_sample_data, generate_drug_response_data

# Set page config
st.set_page_config(
    page_title="Kernel Visualization for Antibiotic Synergy",
    page_icon="🧬",
    layout="wide"
)

# Main title
st.title("Kernel Visualization & Analysis Tool")
st.subheader("For Antibiotic Synergy Research")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", [
    "What are Kernels?", 
    "Kernel Types", 
    "Interactive Visualizations",
    "Kernel Matrices", 
    "Upload Your Data"
])

# Add repository information
st.sidebar.markdown("---")
st.sidebar.markdown("### Project Info")
st.sidebar.info(
    "An educational tool for understanding different kernel types "
    "with applications in antibiotic synergy research."
)

# What are Kernels page
if page == "What are Kernels?":
    st.header("What are Kernels?")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Definition
        A kernel is a mathematical function that measures the similarity between pairs of data points. 
        It acts as a "bridge" to analyze complex relationships in data without explicitly computing 
        coordinates in high-dimensional space.
        
        ### Key Idea
        Kernels allow you to work in a higher-dimensional space while doing all computations in the original 
        input space. This avoids the "curse of dimensionality" and makes complex problems computationally tractable.
        
        ### Why Kernels Matter for Antibiotic Synergy Research
        
        #### 1. Handling Non-Euclidean Relationships
        Biological systems (e.g., bacterial stress-response networks) often have relationships that can't be 
        captured by standard "flat" (Euclidean) geometry.
        
        **Example**: Two antibiotics might disrupt different pathways, but their combined effect creates a 
        non-linear interaction (e.g., one drug amplifies the stress caused by the other). Kernels can model 
        these interactions.
        
        #### 2. Indefinite Similarity Measures
        Traditional kernels require similarity measures to be positive definite (always non-negative). 
        Kreĭn-space kernels relax this constraint, allowing indefinite similarities.
        
        **Why this matters**: Bacterial stress responses might involve conflicting signals (e.g., oxidative 
        stress vs. nutrient deprivation). Kreĭn kernels can model these opposing dynamics.
        
        #### 3. Interpretability
        Kernel methods (like SVMs) produce models where predictions depend on weighted combinations of 
        similarities to training examples. This makes it easier to trace which biological features drive predictions.
        """)
    
    with col2:
        st.markdown("### The Kernel Trick")
        
        # Simple illustration of kernel trick
        x = np.linspace(-5, 5, 100)
        y1 = np.where(x < 0, 1, -1)
        y2 = np.where(x**2 < 5, 1, -1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6))
        
        # Linear case (not separable)
        ax1.scatter(x, np.zeros_like(x), c=y1, cmap='coolwarm', s=20)
        ax1.set_title('Linear Space (Not Separable)')
        ax1.set_ylim(-0.5, 0.5)
        ax1.set_xlabel('x')
        ax1.set_yticks([])
        
        # Transformed with kernel (separable)
        ax2.scatter(x, x**2, c=y2, cmap='coolwarm', s=20)
        ax2.set_title('Kernel-Transformed Space (Separable)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('x²')
        
        st.pyplot(fig)
        
        st.markdown("""
        The "kernel trick" transforms data into a higher-dimensional space where complex patterns 
        become more apparent and easier to separate.
        """)

# Kernel Types page
elif page == "Kernel Types":
    st.header("Types of Kernels")
    
    st.markdown("""
    Different kernel functions capture different types of similarities between data points.
    Below are some common kernels relevant to antibiotic synergy research:
    """)
    
    kernel_tabs = st.tabs(["Linear Kernel", "RBF Kernel", "Graph Kernel", "Kreĭn-Space Kernel"])
    
    with kernel_tabs[0]:
        st.markdown("""
        ### Linear Kernel
        
        **Formula**: $K(x, y) = x^T y$
        
        **Properties**:
        - Simplest kernel function
        - Equivalent to dot product in the input space
        - No transformation to higher dimensions
        
        **Use Case in Antibiotic Research**:
        - Baseline for comparing simple relationships (e.g., additive drug effects)
        - When effects are expected to be proportional to concentrations
        - For initial exploration of data
        """)
        
        # Simple visualization
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = X * Y  # Linear kernel: dot product
        
        fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y)])
        fig.update_layout(
            title='Linear Kernel Visualization',
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='K(x,y)'
            ),
            width=600,
            height=500
        )
        st.plotly_chart(fig)
    
    with kernel_tabs[1]:
        st.markdown("""
        ### Radial Basis Function (RBF) Kernel
        
        **Formula**: $K(x, y) = exp(-\\gamma ||x - y||^2)$
        
        **Properties**:
        - Measures similarity based on distance
        - Maps to infinite-dimensional space
        - Controlled by $\\gamma$ parameter (width)
        
        **Use Case in Antibiotic Research**:
        - Capturing non-linear similarities in omics data (e.g., transcriptomic profiles)
        - Modeling complex drug-target interactions
        - Clustering similar antibiotic response patterns
        """)
        
        # RBF visualization
        gamma = st.slider("Gamma parameter", 0.1, 2.0, 0.5, 0.1)
        
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.exp(-gamma * (X**2 + Y**2))
        
        fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y)])
        fig.update_layout(
            title=f'RBF Kernel Visualization (γ={gamma})',
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='K(x,y)'
            ),
            width=600,
            height=500
        )
        st.plotly_chart(fig)
    
    with kernel_tabs[2]:
        st.markdown("""
        ### Graph Kernel
        
        **Concept**: Measures similarity between graphs representing biological networks
        
        **Properties**:
        - Compares substructures within graphs
        - Can capture pathway similarities
        - Various types: random walk, shortest path, etc.
        
        **Use Case in Antibiotic Research**:
        - Modeling pathway crosstalk as networks (e.g., protein-protein interaction networks)
        - Comparing drug effects on biological network perturbations
        - Analyzing pathway disruption patterns
        """)
        
        # Simple graph visualization
        st.image("https://miro.medium.com/max/1400/1*1wjSdB-FKC2_hFkwzBCuDQ.jpeg", 
                 caption="Example of Graph Kernel Concept (Source: Medium)")
        
        st.markdown("""
        *Note: Graph kernels compare similarities between network structures, which is particularly useful 
        when modeling how antibiotics affect cellular pathways and their interactions.*
        """)
    
    with kernel_tabs[3]:
        st.markdown("""
        ### Kreĭn-Space Kernel
        
        **Concept**: Extends traditional kernels to handle indefinite similarities
        
        **Properties**:
        - Works in spaces with indefinite inner products
        - Can represent both positive and negative similarities
        - Generalizes traditional positive definite kernels
        
        **Use Case in Antibiotic Research**:
        - Handling indefinite similarities in dynamic systems (e.g., conflicting stress-response signals)
        - Modeling antagonistic drug interactions
        - Representing competing biological pathways
        """)
        
        st.markdown("""
        #### Kreĭn Kernels vs. Traditional Kernels
        
        | Feature | Traditional Kernel (e.g., RBF) | Kreĭn-Space Kernel |
        |---------|--------------------------------|-------------------|
        | Similarity Type | Positive definite | Indefinite (can handle negative/"conflicting" similarities) |
        | Biological Use Case | Static relationships | Dynamic, conflicting stress responses |
        | Math Foundation | Hilbert space | Kreĭn space (allows indefinite inner products) |
        
        *Kreĭn-space methods can represent bacterial stress-response trajectories as dynamic graphs, 
        where nodes are pathways and edges represent crosstalk under drug pressure.*
        """)

# Interactive Visualizations page
elif page == "Interactive Visualizations":
    st.header("Interactive Kernel Visualizations")
    
    st.markdown("""
    This interactive visualization demonstrates how different kernels transform data. 
    Select parameters below to see how kernels map data from the original space to feature space.
    """)
    
    # Create columns for controls and visualization
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Controls")
        
        # Data generation options
        data_type = st.selectbox(
            "Select data type",
            ["Two Clusters", "Circles", "Moons", "Linearly Separable", "Drug Response"]
        )
        
        n_samples = st.slider("Number of samples", 10, 200, 100)
        noise = st.slider("Noise level", 0.0, 0.5, 0.1, 0.05)
        
        # Kernel selection
        kernel_type = st.selectbox(
            "Select kernel type",
            ["Linear", "RBF", "Polynomial", "Sigmoid", "Kreĭn-Space"]
        )
        
        # Kernel parameters
        if kernel_type == "RBF":
            gamma = st.slider("Gamma", 0.01, 2.0, 0.5, 0.01)
        elif kernel_type == "Polynomial":
            degree = st.slider("Degree", 1, 5, 3)
            coef0 = st.slider("Coef0", 0.0, 2.0, 1.0, 0.1)
        elif kernel_type == "Sigmoid":
            gamma = st.slider("Gamma", 0.01, 2.0, 0.5, 0.01)
            coef0 = st.slider("Coef0", 0.0, 2.0, 1.0, 0.1)
        elif kernel_type == "Kreĭn-Space":
            pos_gamma = st.slider("Positive Component Gamma", 0.01, 2.0, 0.5, 0.01)
            neg_weight = st.slider("Negative Component Weight", 0.0, 1.0, 0.3, 0.05)
    
    # Generate data based on selection
    if data_type == "Drug Response":
        X, y = generate_drug_response_data(n_samples, noise)
        data_dim = 2  # We'll just show the first 2 dimensions visually
    else:
        X, y = generate_sample_data(data_type, n_samples, noise)
        data_dim = X.shape[1]
    
    # Apply kernel transformation
    if kernel_type == "Linear":
        kernel_params = {}
        transformed_X = apply_kernel_transformation(X, kernel_type, **kernel_params)
    elif kernel_type == "RBF":
        kernel_params = {"gamma": gamma}
        transformed_X = apply_kernel_transformation(X, kernel_type, **kernel_params)
    elif kernel_type == "Polynomial":
        kernel_params = {"degree": degree, "coef0": coef0}
        transformed_X = apply_kernel_transformation(X, kernel_type, **kernel_params)
    elif kernel_type == "Sigmoid":
        kernel_params = {"gamma": gamma, "coef0": coef0}
        transformed_X = apply_kernel_transformation(X, kernel_type, **kernel_params)
    elif kernel_type == "Kreĭn-Space":
        kernel_params = {"pos_gamma": pos_gamma, "neg_weight": neg_weight}
        transformed_X = apply_kernel_transformation(X, kernel_type, **kernel_params)
    
    with col2:
        # Visualize original and transformed data
        fig = plot_transformed_data(X, transformed_X, y, kernel_type, data_type)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### Interpretation
    
    The visualization above shows:
    - **Left**: Original data in its input space
    - **Right**: Data after kernel transformation (projected to 2D for visualization)
    
    Note how different kernels transform the data in unique ways. In antibiotic research, 
    these transformations can reveal hidden patterns in how drugs interact with biological systems.
    """)

# Kernel Matrices page
elif page == "Kernel Matrices":
    st.header("Kernel Matrices")
    
    st.markdown("""
    A kernel matrix represents pairwise similarities between all data points in your dataset.
    For a dataset with n samples, the kernel matrix K is an n×n matrix where K[i,j] is the 
    similarity between samples i and j according to the chosen kernel function.
    
    In antibiotic synergy research, kernel matrices can reveal patterns in how different drug 
    combinations affect biological pathways.
    """)
    
    # Create columns for controls and visualization
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Controls")
        
        # Data generation options
        data_type = st.selectbox(
            "Select data type",
            ["Two Clusters", "Circles", "Moons", "Linearly Separable", "Drug Response"],
            key="matrix_data_type"
        )
        
        n_samples = st.slider("Number of samples", 10, 50, 20, key="matrix_n_samples")
        noise = st.slider("Noise level", 0.0, 0.5, 0.1, 0.05, key="matrix_noise")
        
        # Kernel selection for comparison
        kernel_type1 = st.selectbox(
            "Kernel 1",
            ["Linear", "RBF", "Polynomial", "Sigmoid"],
            key="matrix_kernel1"
        )
        
        kernel_type2 = st.selectbox(
            "Kernel 2",
            ["Linear", "RBF", "Polynomial", "Sigmoid", "Kreĭn-Space"],
            index=4,
            key="matrix_kernel2"
        )
        
        # Kernel parameters
        if kernel_type1 == "RBF" or kernel_type2 == "RBF":
            gamma = st.slider("Gamma (for RBF)", 0.01, 2.0, 0.5, 0.01, key="matrix_gamma")
        
        if kernel_type1 == "Polynomial" or kernel_type2 == "Polynomial":
            degree = st.slider("Degree (for Polynomial)", 1, 5, 3, key="matrix_degree")
            poly_coef0 = st.slider("Coef0 (for Polynomial)", 0.0, 2.0, 1.0, 0.1, key="matrix_poly_coef0")
        
        if kernel_type1 == "Sigmoid" or kernel_type2 == "Sigmoid":
            sigmoid_gamma = st.slider("Gamma (for Sigmoid)", 0.01, 2.0, 0.5, 0.01, key="matrix_sigmoid_gamma")
            sigmoid_coef0 = st.slider("Coef0 (for Sigmoid)", 0.0, 2.0, 1.0, 0.1, key="matrix_sigmoid_coef0")
        
        if kernel_type1 == "Kreĭn-Space" or kernel_type2 == "Kreĭn-Space":
            pos_gamma = st.slider("Positive Component Gamma (for Kreĭn)", 0.01, 2.0, 0.5, 0.01, key="matrix_krein_gamma")
            neg_weight = st.slider("Negative Component Weight (for Kreĭn)", 0.0, 1.0, 0.3, 0.05, key="matrix_krein_weight")
    
    # Generate data based on selection
    if data_type == "Drug Response":
        X, y = generate_drug_response_data(n_samples, noise)
    else:
        X, y = generate_sample_data(data_type, n_samples, noise)
    
    # Compute kernel matrices
    kernel1_params = {}
    kernel2_params = {}
    
    if kernel_type1 == "RBF":
        kernel1_params["gamma"] = gamma
    elif kernel_type1 == "Polynomial":
        kernel1_params["degree"] = degree
        kernel1_params["coef0"] = poly_coef0
    elif kernel_type1 == "Sigmoid":
        kernel1_params["gamma"] = sigmoid_gamma
        kernel1_params["coef0"] = sigmoid_coef0
    
    if kernel_type2 == "RBF":
        kernel2_params["gamma"] = gamma
    elif kernel_type2 == "Polynomial":
        kernel2_params["degree"] = degree
        kernel2_params["coef0"] = poly_coef0
    elif kernel_type2 == "Sigmoid":
        kernel2_params["gamma"] = sigmoid_gamma
        kernel2_params["coef0"] = sigmoid_coef0
    elif kernel_type2 == "Kreĭn-Space":
        kernel2_params["pos_gamma"] = pos_gamma
        kernel2_params["neg_weight"] = neg_weight
    
    K1 = compute_kernel_matrix(X, kernel_type1, **kernel1_params)
    
    if kernel_type2 == "Kreĭn-Space":
        K2 = compute_krein_kernel_matrix(X, pos_gamma=pos_gamma, neg_weight=neg_weight)
    else:
        K2 = compute_kernel_matrix(X, kernel_type2, **kernel2_params)
    
    with col2:
        # Plot kernel matrices side by side
        fig = plot_kernel_matrix(K1, K2, kernel_type1, kernel_type2, y)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### Interpretation
    
    The above heatmaps represent kernel matrices for two different kernel functions:
    
    - **Left**: Standard kernel matrix showing similarities between samples
    - **Right**: For comparison (could be a Kreĭn-space kernel)
    
    **How to interpret:**
    - Bright spots indicate high similarity between samples
    - Dark spots indicate low similarity
    - In Kreĭn-space kernels, negative similarities (blue) can represent conflicting or antagonistic relationships
    
    This visualization helps identify patterns in how different data points relate to each other according to 
    different similarity measures.
    """)

# Upload Your Data page
elif page == "Upload Your Data":
    st.header("Upload Your Data")
    
    st.markdown("""
    Upload your own dataset to explore how different kernels transform your data. 
    This can be particularly useful for analyzing your own antibiotic synergy experiments.
    
    **Requirements:**
    - CSV file format
    - Numerical features in columns
    - Optional label/class column
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load and display data
            data = pd.read_csv(uploaded_file)
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            # Ask for feature columns and label column
            all_columns = data.columns.tolist()
            
            # Feature selection
            if len(all_columns) > 2:
                default_features = all_columns[:-1]  # Default: all but last column
            else:
                default_features = all_columns
                
            feature_cols = st.multiselect(
                "Select feature columns",
                options=all_columns,
                default=default_features
            )
            
            # Label selection (optional)
            label_col = st.selectbox(
                "Select label column (optional)",
                options=["None"] + all_columns,
                index=0
            )
            
            # Process data
            if feature_cols:
                X = data[feature_cols].values
                
                if label_col != "None" and label_col in data.columns:
                    y = data[label_col].values
                else:
                    y = np.zeros(X.shape[0])  # Dummy labels if not provided
                
                # Apply kernel transformation
                st.subheader("Kernel Selection")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    kernel_type = st.selectbox(
                        "Select kernel type",
                        ["Linear", "RBF", "Polynomial", "Sigmoid", "Kreĭn-Space"],
                        key="upload_kernel_type"
                    )
                    
                    # Kernel parameters
                    kernel_params = {}
                    
                    if kernel_type == "RBF":
                        gamma = st.slider("Gamma", 0.01, 2.0, 0.5, 0.01, key="upload_gamma")
                        kernel_params["gamma"] = gamma
                    elif kernel_type == "Polynomial":
                        degree = st.slider("Degree", 1, 5, 3, key="upload_degree")
                        coef0 = st.slider("Coef0", 0.0, 2.0, 1.0, 0.1, key="upload_coef0")
                        kernel_params["degree"] = degree
                        kernel_params["coef0"] = coef0
                    elif kernel_type == "Sigmoid":
                        gamma = st.slider("Gamma", 0.01, 2.0, 0.5, 0.01, key="upload_sigmoid_gamma")
                        coef0 = st.slider("Coef0", 0.0, 2.0, 1.0, 0.1, key="upload_sigmoid_coef0")
                        kernel_params["gamma"] = gamma
                        kernel_params["coef0"] = coef0
                    elif kernel_type == "Kreĭn-Space":
                        pos_gamma = st.slider("Positive Component Gamma", 0.01, 2.0, 0.5, 0.01, key="upload_krein_gamma")
                        neg_weight = st.slider("Negative Component Weight", 0.0, 1.0, 0.3, 0.05, key="upload_krein_weight")
                        kernel_params = {"pos_gamma": pos_gamma, "neg_weight": neg_weight}
                
                with col2:
                    viz_type = st.radio(
                        "Visualization type",
                        ["Kernel Matrix", "Transformed Data"]
                    )
                
                # Compute and visualize
                if viz_type == "Kernel Matrix":
                    st.subheader("Kernel Matrix Visualization")
                    
                    if kernel_type == "Kreĭn-Space":
                        K = compute_krein_kernel_matrix(X, **kernel_params)
                    else:
                        K = compute_kernel_matrix(X, kernel_type, **kernel_params)
                    
                    # Display kernel matrix
                    fig = plt.figure(figsize=(8, 6))
                    plt.imshow(K, cmap='viridis')
                    plt.colorbar(label='Similarity')
                    plt.title(f'{kernel_type} Kernel Matrix')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Add download button for kernel matrix
                    buffer = io.BytesIO()
                    np.savetxt(buffer, K, delimiter=',')
                    buffer.seek(0)
                    st.download_button(
                        label="Download Kernel Matrix as CSV",
                        data=buffer,
                        file_name="kernel_matrix.csv",
                        mime="text/csv"
                    )
                    
                else:  # Transformed Data
                    st.subheader("Transformed Data Visualization")
                    
                    if X.shape[1] > 2:
                        st.info("Your data has more than 2 dimensions. Showing a 2D projection using t-SNE.")
                    
                    if kernel_type == "Kreĭn-Space":
                        K = compute_krein_kernel_matrix(X, **kernel_params)
                    else:
                        K = compute_kernel_matrix(X, kernel_type, **kernel_params)
                    
                    # Use t-SNE to project the kernel matrix to 2D for visualization
                    tsne = TSNE(n_components=2, random_state=42)
                    K_embedded = tsne.fit_transform(K)
                    
                    # Display scatter plot
                    fig = px.scatter(
                        x=K_embedded[:, 0], 
                        y=K_embedded[:, 1],
                        color=y if label_col != "None" else None,
                        labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
                        title=f'Kernel-transformed Data ({kernel_type})'
                    )
                    st.plotly_chart(fig)
            else:
                st.warning("Please select at least one feature column")
        
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.info("Make sure your CSV file contains numerical data and is properly formatted.")
