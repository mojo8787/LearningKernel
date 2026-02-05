import streamlit as st
import numpy as np
import pandas as pd
import datetime
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from kernel_utils import (
    compute_kernel_matrix,
    compute_krein_kernel_matrix,
    apply_kernel_transformation,
    apply_kernel_pca
)
from data_generator import generate_sample_data
from visualization import (
    plot_kernel_matrix,
    plot_transformed_data,
    plot_eigenspectrum,
    plot_synergy_matrix
)
from database import (
    init_db,
    get_all_antibiotics,
    get_all_drug_pairs,
    get_drug_pairs_as_dataframe,
    get_all_analysis_results,
    get_analysis_result_by_id,
    save_analysis_result,
    seed_database
)

# Initialize database
init_db()

# Set page configuration
st.set_page_config(
    page_title="KreinSynergy: Interactive Exploration of Indefinite Kernels",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define helper functions
def get_class_labels(data_type):
    """Return class labels for different data types"""
    if data_type == "moons" or data_type == "circles" or data_type == "linearly_separable":
        return ["Class 0", "Class 1"]
    elif data_type == "drug_response":
        return ["Antagonistic", "Additive", "Synergistic"]
    elif data_type == "bacteria_markers":
        return ["Sensitive", "Intermediate", "Resistant"]
    else:
        return ["Class " + str(i) for i in range(10)]

def get_feature_names(data_type):
    """Return feature names for different data types"""
    if data_type == "moons" or data_type == "circles" or data_type == "linearly_separable":
        return ["Feature 1", "Feature 2"]
    elif data_type == "drug_response":
        return ["Drug A Conc.", "Drug B Conc.", "Target Inhibition", "Stress Response", "Energy Metabolism"]
    elif data_type == "bacteria_markers":
        return ["Drug Target", "Efflux Pump", "Beta-lactamase", "Cell Wall", "Metabolism", "Stress Response"]
    else:
        return ["Feature " + str(i+1) for i in range(10)]

# Create the sidebar
st.sidebar.title("KreinSynergy")
# Using a different image that doesn't require external access
st.sidebar.markdown("## Interactive Exploration of Indefinite Kernels")

# Navigation
st.sidebar.markdown("## Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["Home", "Kernel Types Overview", "Kernel Visualization", 
     "Kreĭn Space Mathematics", "Biological Context", 
     "Experimental Data", "Save & Load Analysis"]
)

# Page: Home
if page == "Home":
    st.title("KreinSynergy: Interactive Exploration of Indefinite Kernels for Antibiotic Synergy Research")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to KreinSynergy

        This interactive application demonstrates the power of Kreĭn space kernels for analyzing complex biological data, 
        with a focus on antibiotic synergy research. The application bridges the gap between machine learning theory and 
        microbiology applications.
        
        ### What are Kreĭn Space Kernels?
        
        Kreĭn space kernels are a class of indefinite kernels that can represent both similarity and dissimilarity 
        relationships in data. Unlike traditional positive definite kernels, Kreĭn kernels can model complex relationships 
        in biological systems where both cooperation and antagonism exist simultaneously.
        
        ### Connection to Antibiotic Synergy
        
        Antibiotic synergy occurs when the combined effect of two antibiotics is greater than the sum of their individual 
        effects. This phenomenon is crucial for treating resistant bacterial infections, but the underlying mechanisms are 
        complex and not fully understood. Kreĭn kernels provide a mathematical framework to model these complex interactions.
        
        ### Application Areas:
        
        - **Drug Combination Optimization**: Finding optimal antibiotic combinations for treating resistant infections
        - **Mechanism Understanding**: Exploring underlying biochemical mechanisms of drug synergy
        - **Resistance Pattern Analysis**: Identifying patterns in bacterial resistance to multiple antibiotics
        - **Drug Discovery**: Guiding the development of new antibiotics designed to work synergistically
        
        ### About This Project
        
        This application was created as part of a research project exploring the applications of advanced kernel methods 
        in microbiological research, specifically focusing on the work of Professor Thomas Gärtner on indefinite kernels
        and their biological applications.
        """)
    
    with col2:
        st.markdown("""
        ### Quick Navigation
        
        - [Kernel Types Overview](#kernel-types-overview)
        - [Kernel Visualization](#kernel-visualization)
        - [Kreĭn Space Mathematics](#kreĭn-space-mathematics)
        - [Biological Context](#biological-context)
        - [Experimental Data](#experimental-data)
        - [Save & Load Analysis](#save-load-analysis)
        
        ### About the Creator
        
        This tool was developed by a microbiologist with an interest in applying advanced computational methods to 
        biological problems. As a potential postdoc candidate, this project demonstrates interdisciplinary skills 
        bridging microbiology and machine learning.
        
        ### Learn More
        
        For more information on Kreĭn space kernels and their applications in computational biology, check out:
        
        - [Thomas Gärtner's Research](https://ml-tuw.github.io/people/thomas-gaertner/)
        - [Learning with Indefinite Kernels](https://doi.org/10.1109/TPAMI.2012.189)
        - [Learning with Non-positive Kernels](https://doi.org/10.1145/1015330.1015443)
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Professor Thomas Gärtner's Research and Its Relevance to Antibiotic Synergy
    
    Professor Thomas Gärtner is a leading researcher in the field of machine learning, with a focus on kernels, graphs, and their applications. His work on indefinite kernels, particularly in Kreĭn spaces, has significant implications for modeling complex biological interactions like antibiotic synergy.
    
    #### Key Research Contributions:
    
    1. **Indefinite Kernel Learning**: Developing methods for learning with indefinite kernels, which is critical for modeling data with both similarity and dissimilarity relationships.
    
    2. **Graph Kernels**: Pioneering work on kernels for structured data like molecular graphs, which allows for sophisticated analysis of antibiotic structures and their interactions.
    
    3. **Kreĭn Space Methods**: Advancing the theoretical understanding of learning in Kreĭn spaces, providing a solid mathematical foundation for analyzing complex biological interactions.
    
    #### Relevance to Antibiotic Synergy Research:
    
    Antibiotic synergy is a complex phenomenon involving multiple biological pathways, molecular interactions, and cellular responses. Traditional positive definite kernels can model similarity relationships but struggle with the antagonistic relationships that are also present in drug interactions. Kreĭn kernels, on the other hand, can naturally represent both types of relationships, making them ideal for modeling antibiotic synergy.
    
    By applying Kreĭn space methods to antibiotic interaction data, researchers can gain new insights into:
    
    - The mechanisms underlying synergistic effects
    - Patterns of cross-resistance and cross-susceptibility
    - Optimal drug combinations for specific bacterial strains
    - Novel targets for antibiotic development
    
    This application aims to bring these advanced mathematical concepts to microbiologists in an accessible, interactive format, facilitating interdisciplinary research in this important area.
    """)

# Page: Kernel Types Overview
elif page == "Kernel Types Overview":
    st.title("Kernel Types Overview")
    
    st.markdown("""
    ### Introduction to Kernels in Machine Learning
    
    Kernel methods are a class of algorithms for pattern analysis, whose best-known member is the Support Vector Machine (SVM).
    The general task of pattern analysis is to find and study general types of relations (e.g. clusters, rankings, principal 
    components, correlations, classifications) in datasets.
    
    A kernel function is a similarity measure between two points. Mathematically, it corresponds to an inner product in a 
    (possibly infinite-dimensional) feature space.
    
    ### Common Kernel Types
    """)
    
    kernel_types = {
        "Linear Kernel": {
            "formula": "K(x, y) = x^T y",
            "description": "The simplest kernel function. It's equivalent to the dot product in the original space. Linear kernels are useful when the data is already linearly separable.",
            "parameters": None,
            "applications": "Document classification, simple regression tasks"
        },
        "Polynomial Kernel": {
            "formula": "K(x, y) = (γx^T y + c)^d",
            "description": "Maps data into a higher-dimensional space where γ, c, and d are parameters. This kernel can capture interactions between features up to degree d.",
            "parameters": "γ (gamma), c (coef0), d (degree)",
            "applications": "Image processing, natural language processing"
        },
        "RBF Kernel (Gaussian)": {
            "formula": "K(x, y) = exp(-γ||x - y||^2)",
            "description": "The Radial Basis Function kernel is one of the most popular kernels. It can map into an infinite-dimensional space and works well for most problems when you're not sure which kernel to use.",
            "parameters": "γ (gamma)",
            "applications": "General purpose, works well for most classification and regression tasks"
        },
        "Sigmoid Kernel": {
            "formula": "K(x, y) = tanh(γx^T y + c)",
            "description": "Also known as the hyperbolic tangent kernel, it is related to neural networks. This kernel is not positive definite for all parameter values.",
            "parameters": "γ (gamma), c (coef0)",
            "applications": "Neural networks, binary classification"
        },
        "Kreĭn Kernel (Indefinite)": {
            "formula": "K(x, y) = K₊(x, y) - K₋(x, y)",
            "description": "Kreĭn kernels are a difference of two positive definite kernels, allowing them to model both similarity and dissimilarity. They can capture complex patterns that positive definite kernels cannot.",
            "parameters": "Parameters for K₊ and K₋, plus the weight for K₋",
            "applications": "Biological networks, drug interaction analysis, anomaly detection"
        }
    }
    
    for kernel_name, kernel_info in kernel_types.items():
        with st.expander(f"{kernel_name}"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"**Formula:** {kernel_info['formula']}")
                
                if kernel_info['parameters']:
                    st.markdown(f"**Parameters:** {kernel_info['parameters']}")
                else:
                    st.markdown("**Parameters:** None")
            
            with col2:
                st.markdown(f"**Description:** {kernel_info['description']}")
                st.markdown(f"**Applications:** {kernel_info['applications']}")
    
    st.markdown("""
    ### Positive Definite vs. Indefinite Kernels
    
    Kernels can be categorized as either positive definite or indefinite:
    
    **Positive Definite Kernels:**
    - The kernel matrix has all non-negative eigenvalues
    - They correspond to an inner product in some Hilbert space
    - Examples: Linear, Polynomial, RBF (Gaussian)
    - Well-behaved mathematically with strong theoretical guarantees
    
    **Indefinite Kernels:**
    - The kernel matrix has both positive and negative eigenvalues
    - They correspond to an inner product in Kreĭn space (difference of two Hilbert spaces)
    - Examples: Sigmoid (for certain parameters), Kreĭn kernels
    - Can capture more complex relationships, but pose theoretical challenges
    
    ### Why Indefinite Kernels Matter for Biological Data
    
    Biological interactions are often complex and involve both cooperative and competitive relationships. For example, in antibiotic synergy:
    
    - Some drug combinations enhance each other's effects (synergy)
    - Some drug combinations reduce each other's effects (antagonism)
    - Some pathways might be promoted while others are inhibited
    
    Indefinite kernels, particularly Kreĭn kernels, can naturally represent these mixed relationships, making them powerful tools for analyzing complex biological systems.
    """)

# Page: Kernel Visualization
elif page == "Kernel Visualization":
    st.title("Kernel Visualization")
    
    # Data generation parameters
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        data_type = st.selectbox(
            "Select data type:",
            ["moons", "circles", "linearly_separable", "drug_response", "bacteria_markers"]
        )
    
    with col2:
        n_samples = st.slider("Number of samples:", min_value=10, max_value=500, value=100, step=10)
    
    with col3:
        noise = st.slider("Noise level:", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
    
    with col4:
        random_state = st.number_input("Random seed:", value=42, min_value=0, max_value=1000, step=1)
    
    # Generate data
    np.random.seed(random_state)
    X, y = generate_sample_data(data_type, n_samples=n_samples, noise=noise)
    
    # Display data info
    st.markdown(f"**Generated {data_type} data with {X.shape[0]} samples, {X.shape[1]} features, and {len(np.unique(y))} classes**")
    
    # Feature names based on data type
    feature_names = get_feature_names(data_type)
    class_labels = get_class_labels(data_type)
    
    # Display original data (if 2D)
    if X.shape[1] <= 2:
        fig = px.scatter(
            x=X[:, 0], 
            y=X[:, 1] if X.shape[1] > 1 else np.zeros(X.shape[0]),
            color=[class_labels[i] for i in y],
            labels={'x': feature_names[0], 'y': feature_names[1] if X.shape[1] > 1 else ''},
            title=f"Original {data_type.capitalize()} Data"
        )
        st.plotly_chart(fig)
    else:
        st.markdown(f"**Note:** Original data has {X.shape[1]} dimensions, displaying transformed views below.")
    
    # Kernel selection
    st.markdown("### Select Kernels to Compare")
    
    col1, col2 = st.columns(2)
    
    with col1:
        kernel1_type = st.selectbox(
            "Kernel 1 type:",
            ["linear", "rbf", "polynomial", "sigmoid", "kreĭn"],
            key="kernel1_type"
        )
        
        # Parameters for kernel 1
        if kernel1_type == "rbf":
            k1_gamma = st.slider("Gamma (kernel 1):", min_value=0.01, max_value=5.0, value=1.0, step=0.1, key="k1_gamma")
            kernel1_params = {"gamma": k1_gamma}
        
        elif kernel1_type == "polynomial":
            k1_degree = st.slider("Degree (kernel 1):", min_value=1, max_value=10, value=3, step=1, key="k1_degree")
            k1_gamma = st.slider("Gamma (kernel 1):", min_value=0.01, max_value=5.0, value=1.0, step=0.1, key="k1_gamma")
            k1_coef0 = st.slider("Coef0 (kernel 1):", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key="k1_coef0")
            kernel1_params = {"degree": k1_degree, "gamma": k1_gamma, "coef0": k1_coef0}
        
        elif kernel1_type == "sigmoid":
            k1_gamma = st.slider("Gamma (kernel 1):", min_value=0.01, max_value=5.0, value=1.0, step=0.1, key="k1_gamma")
            k1_coef0 = st.slider("Coef0 (kernel 1):", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key="k1_coef0")
            kernel1_params = {"gamma": k1_gamma, "coef0": k1_coef0}
        
        elif kernel1_type == "kreĭn":
            k1_pos_gamma = st.slider("Positive component gamma (kernel 1):", min_value=0.01, max_value=5.0, value=0.5, step=0.1, key="k1_pos_gamma")
            k1_neg_weight = st.slider("Negative component weight (kernel 1):", min_value=0.0, max_value=1.0, value=0.3, step=0.05, key="k1_neg_weight")
            kernel1_params = {"pos_gamma": k1_pos_gamma, "neg_weight": k1_neg_weight}
            
        else:  # linear kernel doesn't need parameters
            kernel1_params = {}
    
    with col2:
        kernel2_type = st.selectbox(
            "Kernel 2 type:",
            ["linear", "rbf", "polynomial", "sigmoid", "kreĭn"],
            index=4,  # Default to kreĭn for kernel 2
            key="kernel2_type"
        )
        
        # Parameters for kernel 2
        if kernel2_type == "rbf":
            k2_gamma = st.slider("Gamma (kernel 2):", min_value=0.01, max_value=5.0, value=1.0, step=0.1, key="k2_gamma")
            kernel2_params = {"gamma": k2_gamma}
        
        elif kernel2_type == "polynomial":
            k2_degree = st.slider("Degree (kernel 2):", min_value=1, max_value=10, value=3, step=1, key="k2_degree")
            k2_gamma = st.slider("Gamma (kernel 2):", min_value=0.01, max_value=5.0, value=1.0, step=0.1, key="k2_gamma")
            k2_coef0 = st.slider("Coef0 (kernel 2):", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key="k2_coef0")
            kernel2_params = {"degree": k2_degree, "gamma": k2_gamma, "coef0": k2_coef0}
        
        elif kernel2_type == "sigmoid":
            k2_gamma = st.slider("Gamma (kernel 2):", min_value=0.01, max_value=5.0, value=1.0, step=0.1, key="k2_gamma")
            k2_coef0 = st.slider("Coef0 (kernel 2):", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key="k2_coef0")
            kernel2_params = {"gamma": k2_gamma, "coef0": k2_coef0}
        
        elif kernel2_type == "kreĭn":
            k2_pos_gamma = st.slider("Positive component gamma (kernel 2):", min_value=0.01, max_value=5.0, value=0.5, step=0.1, key="k2_pos_gamma")
            k2_neg_weight = st.slider("Negative component weight (kernel 2):", min_value=0.0, max_value=1.0, value=0.3, step=0.05, key="k2_neg_weight")
            kernel2_params = {"pos_gamma": k2_pos_gamma, "neg_weight": k2_neg_weight}
            
        else:  # linear kernel doesn't need parameters
            kernel2_params = {}
    
    # Compute kernel matrices
    K1 = compute_kernel_matrix(X, kernel1_type, **kernel1_params)
    K2 = compute_kernel_matrix(X, kernel2_type, **kernel2_params)
    
    # Plot kernel matrices
    st.markdown("### Kernel Matrix Comparison")
    
    fig = plot_kernel_matrix(K1, K2, kernel1_type.capitalize(), kernel2_type.capitalize(), y)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display eigenvalue spectrum for both kernels
    st.markdown("### Eigenvalue Spectrum")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = plot_eigenspectrum(K1, kernel1_type.capitalize())
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = plot_eigenspectrum(K2, kernel2_type.capitalize())
        st.plotly_chart(fig, use_container_width=True)
    
    # Apply kernel transformations to data
    st.markdown("### Data Transformation with Kernels")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Transform with kernel 1
        X_transformed_1 = apply_kernel_transformation(X, kernel1_type, **kernel1_params)
        fig = plot_transformed_data(X, X_transformed_1, y, kernel1_type.capitalize(), data_type.capitalize())
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Transform with kernel 2
        X_transformed_2 = apply_kernel_transformation(X, kernel2_type, **kernel2_params)
        fig = plot_transformed_data(X, X_transformed_2, y, kernel2_type.capitalize(), data_type.capitalize())
        st.plotly_chart(fig, use_container_width=True)
    
    # Save analysis button
    st.markdown("### Save This Analysis")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        save_name = st.text_input("Analysis name:", value=f"{data_type}_{kernel1_type}_vs_{kernel2_type}")
    
    with col2:
        save_description = st.text_input("Description:", value=f"Comparison of {kernel1_type} and {kernel2_type} kernels on {data_type} data")
    
    if st.button("Save Analysis"):
        analysis_data = {
            "data_type": data_type,
            "n_samples": n_samples,
            "noise": noise,
            "random_state": random_state,
            "kernel1_type": kernel1_type,
            "kernel1_params": kernel1_params,
            "kernel2_type": kernel2_type,
            "kernel2_params": kernel2_params,
            "X": X.tolist(),
            "y": y.tolist()
        }
        
        created_at = datetime.datetime.now().isoformat()
        
        # Save to database
        result_id = save_analysis_result(
            name=save_name,
            description=save_description,
            kernel_type=f"{kernel1_type}_vs_{kernel2_type}",
            kernel_params={"kernel1": kernel1_params, "kernel2": kernel2_params},
            visualization_data=analysis_data,
            created_at=created_at
        )
        
        st.success(f"Analysis saved with ID: {result_id}")

# Page: Kreĭn Space Mathematics
elif page == "Kreĭn Space Mathematics":
    st.title("Kreĭn Space Mathematics")
    
    st.markdown("""
    ### Understanding Kreĭn Spaces
    
    A Kreĭn space is a special type of vector space that generalizes the concept of Hilbert spaces by allowing for indefinite inner products. 
    This mathematical structure is particularly useful for modeling complex systems with both positive and negative interactions.
    
    #### Definition
    
    A Kreĭn space $K$ can be decomposed as:
    
    $$K = K_+ \oplus K_-$$
    
    where $K_+$ and $K_-$ are Hilbert spaces, and the inner product in $K$ is defined as:
    
    $$\langle x, y \rangle_K = \langle x_+, y_+ \rangle_{K_+} - \langle x_-, y_- \rangle_{K_-}$$
    
    where $x = x_+ + x_-$ and $y = y_+ + y_-$ with $x_+, y_+ \in K_+$ and $x_-, y_- \in K_-$.
    
    #### Kreĭn Kernels
    
    A Kreĭn kernel $\kappa$ is a function that can be decomposed as:
    
    $$\kappa(x, y) = \kappa_+(x, y) - \kappa_-(x, y)$$
    
    where $\kappa_+$ and $\kappa_-$ are positive definite kernels.
    """)
    
    # Interactive demonstration
    st.markdown("### Interactive Demonstration: Building a Kreĭn Kernel")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pos_kernel_type = st.selectbox(
            "Positive Component Type:",
            ["rbf", "polynomial", "linear"],
            index=0
        )
        
        if pos_kernel_type == "rbf":
            pos_gamma = st.slider("Gamma (positive component):", min_value=0.01, max_value=5.0, value=0.5, step=0.1)
            pos_params = {"gamma": pos_gamma}
        
        elif pos_kernel_type == "polynomial":
            pos_degree = st.slider("Degree (positive component):", min_value=1, max_value=10, value=3, step=1)
            pos_gamma = st.slider("Gamma (positive component):", min_value=0.01, max_value=5.0, value=1.0, step=0.1)
            pos_params = {"degree": pos_degree, "gamma": pos_gamma, "coef0": 1.0}
            
        else:  # linear
            pos_params = {}
    
    with col2:
        neg_weight = st.slider("Negative Component Weight:", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
        
        st.markdown("""
        The negative component in this demo uses a simple distance-based measure:
        
        $$\kappa_-(x, y) = -||x - y||^2$$
        
        This captures dissimilarity between points. The weight controls the balance between positive and negative components.
        """)
    
    # Generate example data
    n_demo = 50
    np.random.seed(42)
    X_demo = np.random.rand(n_demo, 2) * 2 - 1  # Random 2D points in [-1, 1] x [-1, 1]
    
    # Compute positive kernel matrix
    K_pos = compute_kernel_matrix(X_demo, pos_kernel_type, **pos_params)
    
    # Compute negative kernel matrix (simplified for demonstration)
    n_samples = X_demo.shape[0]
    K_neg = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(n_samples):
            K_neg[i, j] = -np.sum((X_demo[i] - X_demo[j]) ** 2)
    
    # Normalize K_neg to have similar scale to K_pos
    K_neg = K_neg / np.abs(K_neg).max() if np.abs(K_neg).max() > 0 else K_neg
    
    # Compute Kreĭn kernel
    K_krein = K_pos - neg_weight * K_neg
    
    # Plot
    tab1, tab2, tab3, tab4 = st.tabs(["Kernel Matrices", "Data Transformation", "Eigenvalues", "Mathematical Intuition"])
    
    with tab1:
        # Create figure with 1 row and 3 columns
        fig = make_subplots(rows=1, cols=3, subplot_titles=["Positive Component", "Negative Component", "Kreĭn Kernel"])
        
        # Add heatmap for positive component
        fig.add_trace(
            go.Heatmap(
                z=K_pos,
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Similarity", x=0.3, y=0.8, len=0.5)
            ),
            row=1, col=1
        )
        
        # Add heatmap for negative component
        fig.add_trace(
            go.Heatmap(
                z=K_neg * neg_weight,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Dissimilarity", x=0.65, y=0.8, len=0.5)
            ),
            row=1, col=2
        )
        
        # Add heatmap for Kreĭn kernel
        fig.add_trace(
            go.Heatmap(
                z=K_krein,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Kreĭn Value", x=1.0, y=0.8, len=0.5)
            ),
            row=1, col=3
        )
        
        # Update layout
        fig.update_layout(
            title_text="Kreĭn Kernel Decomposition",
            height=400,
            width=900
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Transform data with the different kernels for visualization
        # Use t-SNE for visualization
        
        # Create artificial classes for visualization (just for coloring)
        y_demo = (X_demo[:, 0] > 0).astype(int) + (X_demo[:, 1] > 0).astype(int)
        
        # Transform with each kernel type
        X_pos_transformed = apply_kernel_transformation(X_demo, pos_kernel_type, **pos_params)
        
        # For negative component, use a custom transformation
        K_neg_norm = K_neg / np.abs(K_neg).max() if np.abs(K_neg).max() > 0 else K_neg
        D_neg = 1 - K_neg_norm
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, metric='precomputed', perplexity=min(30, X_demo.shape[0]-1), init='random')
        X_neg_transformed = tsne.fit_transform(D_neg)
        
        # Transform with Kreĭn kernel
        X_krein_transformed = apply_kernel_transformation(X_demo, "kreĭn", pos_gamma=pos_params.get("gamma", 1.0), neg_weight=neg_weight)
        
        # Create figure
        fig = make_subplots(rows=2, cols=2, 
                           subplot_titles=["Original Data", "Positive Component", "Negative Component", "Kreĭn Kernel"])
        
        # Add scatter for original data
        fig.add_trace(
            go.Scatter(
                x=X_demo[:, 0],
                y=X_demo[:, 1],
                mode='markers',
                marker=dict(size=10, color=y_demo, colorscale='Viridis', showscale=False),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add scatter for positive component
        fig.add_trace(
            go.Scatter(
                x=X_pos_transformed[:, 0],
                y=X_pos_transformed[:, 1],
                mode='markers',
                marker=dict(size=10, color=y_demo, colorscale='Viridis', showscale=False),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add scatter for negative component
        fig.add_trace(
            go.Scatter(
                x=X_neg_transformed[:, 0],
                y=X_neg_transformed[:, 1],
                mode='markers',
                marker=dict(size=10, color=y_demo, colorscale='Viridis', showscale=False),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add scatter for Kreĭn kernel
        fig.add_trace(
            go.Scatter(
                x=X_krein_transformed[:, 0],
                y=X_krein_transformed[:, 1],
                mode='markers',
                marker=dict(size=10, color=y_demo, colorscale='Viridis', showscale=True, 
                           colorbar=dict(title="Class", x=1.0)),
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=700,
            width=900,
            title_text="Data Transformation with Different Kernels"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Note:** The visualizations above show how the same data is transformed differently by each kernel component. 
        The Kreĭn kernel combines the characteristics of both positive and negative components, potentially revealing 
        structures that neither component could capture alone.
        """)
    
    with tab3:
        # Plot eigenvalues of each kernel matrix
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Positive Component")
            fig = plot_eigenspectrum(K_pos, pos_kernel_type.capitalize(), "Eigenspectrum of Positive Component")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Negative Component")
            fig = plot_eigenspectrum(K_neg * neg_weight, "Negative", "Eigenspectrum of Negative Component")
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.markdown("#### Kreĭn Kernel")
            fig = plot_eigenspectrum(K_krein, "Kreĭn", "Eigenspectrum of Kreĭn Kernel")
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Observations:**
        
        1. The positive component has only positive eigenvalues, as expected for a positive definite kernel.
        2. The negative component has only negative eigenvalues (when weighted).
        3. The Kreĭn kernel has a mixture of positive and negative eigenvalues, reflecting its indefinite nature.
        
        This indefiniteness is what gives Kreĭn kernels their power to model complex relationships that cannot be 
        captured by positive definite kernels alone.
        """)
    
    with tab4:
        st.markdown("""
        ### Mathematical Intuition Behind Kreĭn Kernels
        
        To develop intuition for Kreĭn kernels, consider these key concepts:
        
        #### 1. Positive Definite Kernels
        
        A positive definite kernel $\kappa_+$ measures *similarity* between points. Higher values indicate 
        greater similarity. These kernels embed data into a space where similar points are close together.
        
        #### 2. Negative Definite Kernels
        
        A negative definite kernel $\kappa_-$ measures *dissimilarity* between points. Higher values indicate 
        greater dissimilarity. These kernels embed data into a space where dissimilar points are pushed apart.
        
        #### 3. Kreĭn Kernels
        
        A Kreĭn kernel $\kappa = \kappa_+ - \kappa_-$ combines both similarity and dissimilarity measures.
        This allows it to model:
        
        - Points that should be close together (through $\kappa_+$)
        - Points that should be far apart (through $\kappa_-$)
        - Complex relationships where some features indicate similarity while others indicate dissimilarity
        
        #### 4. Biological Relevance
        
        In biological systems, we often encounter:
        
        - Cooperative interactions (synergy) → modeled by $\kappa_+$
        - Competitive interactions (antagonism) → modeled by $\kappa_-$
        - Complex interactions involving both → modeled by the full Kreĭn kernel $\kappa$
        
        For example, in antibiotic synergy, two drugs might:
        - Target similar pathways (similarity)
        - Have opposing mechanisms in other pathways (dissimilarity)
        - Result in either synergy or antagonism depending on which effect dominates
        
        Kreĭn kernels provide a natural mathematical framework to model these complex biological phenomena.
        """)
        
        st.markdown("""
        **Mark Krein (1907-1989)** was a mathematician who contributed significantly 
        to the theory of indefinite inner product spaces. His work laid the foundation 
        for many modern applications in mathematical physics and machine learning.
        """)
        
        st.markdown("""
        **Further Reading:**
        
        - Ong, C.S., Mary, X., Canu, S., Smola, A.J.: Learning with non-positive kernels. In: Proceedings of the Twenty-First International Conference on Machine Learning (2004)
        
        - Haasdonk, B.: Feature space interpretation of SVMs with indefinite kernels. IEEE Transactions on Pattern Analysis and Machine Intelligence (2005)
        
        - Gärtner, T., Lloyd, J.W., Flach, P.A.: Kernels and distances for structured data. Machine Learning (2004)
        """)

# Page: Biological Context
elif page == "Biological Context":
    st.title("Biological Context: Antibiotic Synergy")
    
    st.markdown("""
    ### What is Antibiotic Synergy?
    
    Antibiotic synergy occurs when the combined effect of two antibiotics is greater than the sum of their individual effects. 
    This phenomenon is crucial for designing effective treatments for bacterial infections, especially in cases of antimicrobial resistance.
    
    #### Key Concepts:
    
    - **Fractional Inhibitory Concentration Index (FICI)**: A measure of drug interaction
      - FICI < 0.5: Synergy
      - 0.5 ≤ FICI ≤ 4: Additivity/Indifference
      - FICI > 4: Antagonism
    
    - **Mechanisms of Synergy**:
      - Inhibition of different steps in the same biochemical pathway
      - Enhanced penetration of one drug due to the action of another
      - Inhibition of resistance mechanisms by one drug enabling another drug's action
      - Interaction with different targets that results in enhanced bacterial killing
    
    - **Significance in Clinical Settings**:
      - Combating antibiotic resistance
      - Reducing individual drug dosages
      - Minimizing side effects
      - Expanding the spectrum of activity
    """)
    
    # Interactive visualization
    st.markdown("### Interactive Visualization of Drug Interactions")
    
    # Drug interaction model
    st.markdown("""
    Below is a simplified model of drug interactions based on their mechanisms of action. 
    You can adjust the parameters to see how different factors influence synergy.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        drug_a_conc = st.slider("Drug A concentration:", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        cell_wall_inhib = st.slider("Cell wall inhibition effect:", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    
    with col2:
        drug_b_conc = st.slider("Drug B concentration:", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        protein_synth_inhib = st.slider("Protein synthesis inhibition:", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    
    with col3:
        penetration_factor = st.slider("Permeability enhancement:", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
        resistance_suppression = st.slider("Resistance mechanism suppression:", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    
    # Calculate synergy score
    # Simple model: synergy scores based on provided factors
    independent_effect = drug_a_conc * cell_wall_inhib + drug_b_conc * protein_synth_inhib
    synergy_contribution = (drug_a_conc * drug_b_conc * penetration_factor) + (resistance_suppression * max(drug_a_conc, drug_b_conc))
    
    # Rescale to 0-1
    max_possible = 2 + 1  # max of independent effect + max of synergy contribution
    combined_effect = (independent_effect + synergy_contribution) / max_possible
    
    # Calculate FICI (simplified model)
    # Lower FICI indicates stronger synergy
    fici = 1.0 - synergy_contribution  # Simplified: higher synergy contribution gives lower FICI
    
    # Classify the interaction
    if fici < 0.5:
        interaction_type = "Synergistic"
        interaction_color = "green"
    elif fici <= 4:
        interaction_type = "Additive/Indifferent"
        interaction_color = "blue"
    else:
        interaction_type = "Antagonistic"
        interaction_color = "red"
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Combined Effect", f"{combined_effect:.2f}")
    
    with col2:
        st.metric("FICI", f"{fici:.2f}")
    
    with col3:
        st.markdown(f"<h3 style='color:{interaction_color};'>Interaction: {interaction_type}</h3>", unsafe_allow_html=True)
    
    # Visualization of drug interaction
    st.markdown("### Visualization of Drug Combination Effects")
    
    # Create a grid of drug concentrations
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x, y)
    
    # Calculate combined effect for each combination
    Z = np.zeros_like(X)
    FICI = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            drug_a = X[i, j]
            drug_b = Y[i, j]
            indep_effect = drug_a * cell_wall_inhib + drug_b * protein_synth_inhib
            syn_contrib = (drug_a * drug_b * penetration_factor) + (resistance_suppression * max(drug_a, drug_b))
            Z[i, j] = (indep_effect + syn_contrib) / max_possible
            FICI[i, j] = 1.0 - syn_contrib
    
    # Create classification matrix
    classes = np.zeros_like(FICI, dtype=int)
    classes[FICI < 0.5] = 2  # Synergistic
    classes[(FICI >= 0.5) & (FICI <= 4)] = 1  # Additive
    classes[FICI > 4] = 0  # Antagonistic
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Combined Effect", "FICI", "Interaction Class"])
    
    with tab1:
        fig = go.Figure(data=[
            go.Surface(z=Z, x=X, y=Y, colorscale='Viridis',
                      colorbar=dict(title="Combined Effect"))
        ])
        
        fig.update_layout(
            title="Combined Effect of Drug A and Drug B",
            scene=dict(
                xaxis_title="Drug A Concentration",
                yaxis_title="Drug B Concentration",
                zaxis_title="Combined Effect",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                zaxis=dict(range=[0, 1])
            ),
            width=700,
            height=700
        )
        
        # Add a point for the current selection
        fig.add_trace(
            go.Scatter3d(
                x=[drug_a_conc],
                y=[drug_b_conc],
                z=[combined_effect],
                mode='markers',
                marker=dict(
                    size=10,
                    color='red',
                ),
                name="Current Selection"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = go.Figure(data=[
            go.Surface(z=FICI, x=X, y=Y, colorscale='Viridis',
                      colorbar=dict(title="FICI"))
        ])
        
        fig.update_layout(
            title="FICI Value for Drug A and Drug B Combinations",
            scene=dict(
                xaxis_title="Drug A Concentration",
                yaxis_title="Drug B Concentration",
                zaxis_title="FICI",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1])
            ),
            width=700,
            height=700
        )
        
        # Add a point for the current selection
        fig.add_trace(
            go.Scatter3d(
                x=[drug_a_conc],
                y=[drug_b_conc],
                z=[fici],
                mode='markers',
                marker=dict(
                    size=10,
                    color='red',
                ),
                name="Current Selection"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Create a 2D heatmap of interaction classes
        fig = go.Figure(data=go.Heatmap(
            z=classes,
            x=x,
            y=y,
            colorscale=[
                [0, 'red'],
                [0.5, 'blue'],
                [1, 'green']
            ],
            showscale=True,
            colorbar=dict(
                title="Interaction Class",
                tickvals=[0, 1, 2],
                ticktext=["Antagonistic", "Additive", "Synergistic"]
            )
        ))
        
        fig.update_layout(
            title="Drug Interaction Classification",
            xaxis_title="Drug A Concentration",
            yaxis_title="Drug B Concentration",
            width=700,
            height=700
        )
        
        # Add a point for the current selection
        fig.add_trace(
            go.Scatter(
                x=[drug_a_conc],
                y=[drug_b_conc],
                mode='markers',
                marker=dict(
                    size=15,
                    color='white',
                    line=dict(
                        color='black',
                        width=2
                    )
                ),
                name="Current Selection"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### Connection to Kreĭn Space Kernels
    
    The complex interactions between antibiotics demonstrate why Kreĭn space kernels are valuable for modeling antibiotic synergy:
    
    1. **Dual Nature of Interactions**: Antibiotics can interact both synergistically and antagonistically, which aligns with the positive and negative components of Kreĭn kernels.
    
    2. **Multiple Interaction Mechanisms**: Various mechanisms (cell wall disruption, protein synthesis inhibition, etc.) contribute differently to the overall effect. Kreĭn kernels can model these complex relationships.
    
    3. **Non-linear Responses**: Drug interactions often produce non-linear responses that cannot be captured by simple additive models. The indefinite nature of Kreĭn kernels allows for modeling such non-linearities.
    
    4. **Biological Pathway Interactions**: Some pathways enhance each other (positive interactions) while others interfere (negative interactions). Kreĭn kernels naturally represent this duality.
    
    By using Kreĭn kernels to analyze antibiotic combination data, researchers can uncover patterns that might be missed by traditional methods, potentially leading to the discovery of novel synergistic combinations for treating resistant infections.
    """)

# Page: Experimental Data
elif page == "Experimental Data":
    st.title("Experimental Data Analysis")
    
    st.markdown("""
    ### Antibiotic Synergy Dataset
    
    This section demonstrates the application of kernel methods to real antibiotic synergy data. 
    The dataset contains information about various antibiotic combinations and their synergistic effects.
    """)
    
    # Get data from the database
    drug_pairs_df = get_drug_pairs_as_dataframe()
    
    # Display the data
    st.dataframe(drug_pairs_df)
    
    # Plot synergy matrix (only if we have data)
    st.markdown("### Antibiotic Synergy Matrix")
    
    if len(drug_pairs_df) > 0:
        fig = plot_synergy_matrix(drug_pairs_df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No drug pair data available. Run `python init_db.py` to seed the database with sample data.")
    
    # Kernel-based analysis
    st.markdown("### Kernel-Based Analysis of Synergy Patterns")
    
    st.markdown("""
    We can apply various kernel methods to analyze patterns in the synergy data. This helps identify 
    antibiotics with similar synergy profiles or discover clusters of drug combinations with similar effects.
    """)
    
    # Prepare data for kernel analysis (requires at least 1 drug pair)
    drugs = sorted(list(set(drug_pairs_df['Drug_A'].unique()) | set(drug_pairs_df['Drug_B'].unique())))
    n_drugs = len(drugs)
    
    if n_drugs < 2:
        st.warning("Kernel-based analysis requires at least 2 antibiotics in the dataset. Run `python init_db.py` to seed the database with sample data.")
    else:
        # Create drug to index mapping
        drug_to_idx = {drug: i for i, drug in enumerate(drugs)}
        
        # Create synergy matrix
        synergy_matrix = np.zeros((n_drugs, n_drugs))
        
        # Fill the matrix
        for _, row in drug_pairs_df.iterrows():
            i = drug_to_idx[row['Drug_A']]
            j = drug_to_idx[row['Drug_B']]
            synergy_matrix[i, j] = row['Synergy_Score']
            synergy_matrix[j, i] = row['Synergy_Score']  # Make it symmetric
        
        # Feature matrix: each row is a drug, with its synergy profile as features
        X_synergy = synergy_matrix
        
        # Select kernels for analysis
        col1, col2 = st.columns(2)
        
        with col1:
            kernel_type = st.selectbox(
                "Select kernel type for analysis:",
                ["linear", "rbf", "polynomial", "kreĭn"],
                index=3
            )
        
        with col2:
            # Parameters based on selected kernel
            if kernel_type == "rbf":
                gamma = st.slider("Gamma parameter:", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                kernel_params = {"gamma": gamma}
            elif kernel_type == "polynomial":
                degree = st.slider("Polynomial degree:", min_value=1, max_value=5, value=3, step=1)
                kernel_params = {"degree": degree, "gamma": 1.0, "coef0": 1.0}
            elif kernel_type == "kreĭn":
                pos_gamma = st.slider("Positive component gamma:", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                neg_weight = st.slider("Negative component weight:", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
                kernel_params = {"pos_gamma": pos_gamma, "neg_weight": neg_weight}
            else:  # linear
                kernel_params = {}
        
        # Compute kernel matrix
        K = compute_kernel_matrix(X_synergy, kernel_type, **kernel_params)
        
        # Apply kernel transformation for visualization
        try:
            X_transformed = apply_kernel_transformation(X_synergy, kernel_type, **kernel_params)
            
            # Create a figure
            fig = go.Figure()
            
            # Add scatter plot
            fig.add_trace(
                go.Scatter(
                    x=X_transformed[:, 0],
                    y=X_transformed[:, 1],
                    mode='markers+text',
                    marker=dict(size=15, color=np.arange(len(drugs)), colorscale='Viridis', showscale=False),
                    text=drugs,
                    textposition="top center",
                    hoverinfo="text"
                )
            )
            
            # Update layout
            fig.update_layout(
                title=f"Antibiotics Mapped with {kernel_type.capitalize()} Kernel",
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                height=600,
                width=800
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Interpretation:** 
            
            The plot above shows antibiotics mapped to a 2D space based on their synergy profiles. 
            Antibiotics that are close together in this space have similar synergy patterns with other antibiotics, 
            suggesting they might have similar mechanisms of action or interact with similar pathways.
            
            This visualization can help identify:
            - Clusters of antibiotics with similar synergy behavior
            - Outliers that have unique synergy profiles
            - Potential substitutions for antibiotics in combination therapies
            """)
            
            # Display eigenspectrum
            st.markdown("### Eigenspectrum Analysis")
            
            fig = plot_eigenspectrum(K, kernel_type.capitalize())
            st.plotly_chart(fig, use_container_width=True)
            
            if kernel_type == "kreĭn":
                st.markdown("""
                The presence of both positive and negative eigenvalues in the Kreĭn kernel's eigenspectrum 
                indicates that the synergy data contains both cooperative and competitive relationships. 
                This indefinite structure can reveal complex patterns that positive definite kernels might miss.
                """)
            else:
                st.markdown("""
                The eigenspectrum above shows the distribution of eigenvalues for the kernel matrix. 
                The magnitude of eigenvalues indicates the importance of different dimensions in the data.
                """)
        
        except Exception as e:
            st.error(f"Error in kernel transformation: {e}")
            st.markdown("""
            The transformation couldn't be computed. This might be due to:
            - Numerical instability in the kernel matrix
            - Insufficient data for a meaningful transformation
            - Issues with the selected parameters
            
            Try adjusting the kernel parameters or selecting a different kernel type.
            """)

# Page: Save & Load Analysis
elif page == "Save & Load Analysis":
    st.title("Save & Load Analysis Results")
    
    # Create tabs for saving and loading
    save_tab, load_tab = st.tabs(["Save New Analysis", "Load Saved Analysis"])
    
    with save_tab:
        st.markdown("""
        ### Save Your Analysis
        
        Save your current analysis settings and results to the database for future reference.
        """)
        
        # Define data generation parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            data_type = st.selectbox(
                "Select data type:",
                ["moons", "circles", "linearly_separable", "drug_response", "bacteria_markers"]
            )
            
            n_samples = st.slider("Number of samples:", min_value=10, max_value=500, value=100, step=10)
        
        with col2:
            kernel_type = st.selectbox(
                "Select kernel type:",
                ["linear", "rbf", "polynomial", "sigmoid", "kreĭn"]
            )
            
            # Parameters based on kernel type
            if kernel_type == "rbf":
                gamma = st.slider("Gamma:", min_value=0.01, max_value=5.0, value=1.0, step=0.1)
                kernel_params = {"gamma": gamma}
            
            elif kernel_type == "polynomial":
                degree = st.slider("Degree:", min_value=1, max_value=10, value=3, step=1)
                gamma = st.slider("Gamma:", min_value=0.01, max_value=5.0, value=1.0, step=0.1)
                coef0 = st.slider("Coef0:", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
                kernel_params = {"degree": degree, "gamma": gamma, "coef0": coef0}
            
            elif kernel_type == "sigmoid":
                gamma = st.slider("Gamma:", min_value=0.01, max_value=5.0, value=1.0, step=0.1)
                coef0 = st.slider("Coef0:", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
                kernel_params = {"gamma": gamma, "coef0": coef0}
            
            elif kernel_type == "kreĭn":
                pos_gamma = st.slider("Positive component gamma:", min_value=0.01, max_value=5.0, value=0.5, step=0.1)
                neg_weight = st.slider("Negative component weight:", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
                kernel_params = {"pos_gamma": pos_gamma, "neg_weight": neg_weight}
                
            else:  # linear kernel
                kernel_params = {}
        
        with col3:
            noise = st.slider("Noise level:", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
            random_state = st.number_input("Random seed:", value=42, min_value=0, max_value=1000, step=1)
            
            st.markdown("### Analysis Information")
            analysis_name = st.text_input("Analysis name:", value=f"{data_type}_{kernel_type}_analysis")
            analysis_desc = st.text_area("Description:", value=f"Analysis of {data_type} data using {kernel_type} kernel")
        
        # Generate data
        np.random.seed(random_state)
        X, y = generate_sample_data(data_type, n_samples=n_samples, noise=noise)
        
        # Compute kernel matrix
        K = compute_kernel_matrix(X, kernel_type, **kernel_params)
        
        # Apply transformation for visualization
        X_transformed = apply_kernel_transformation(X, kernel_type, **kernel_params)
        
        # Plot transformed data
        fig = plot_transformed_data(X, X_transformed, y, kernel_type.capitalize(), data_type.capitalize())
        st.plotly_chart(fig, use_container_width=True)
        
        # Save button
        if st.button("Save Analysis to Database"):
            # Prepare data for saving
            visualization_data = {
                "data_type": data_type,
                "n_samples": n_samples,
                "noise": noise,
                "random_state": random_state,
                "X": X.tolist(),
                "y": y.tolist(),
                "X_transformed": X_transformed.tolist()
            }
            
            # Current datetime
            created_at = datetime.datetime.now().isoformat()
            
            # Save to database
            result_id = save_analysis_result(
                name=analysis_name,
                description=analysis_desc,
                kernel_type=kernel_type,
                kernel_params=kernel_params,
                visualization_data=visualization_data,
                created_at=created_at
            )
            
            st.success(f"Analysis saved with ID: {result_id}")
    
    with load_tab:
        st.markdown("""
        ### Load Saved Analysis
        
        View and explore previously saved analysis results.
        """)
        
        # Get all saved analyses
        analyses = get_all_analysis_results()
        
        if not analyses:
            st.info("No saved analyses found. Create and save an analysis first.")
        else:
            # Convert to DataFrame for display
            analyses_data = []
            for analysis in analyses:
                analyses_data.append({
                    "ID": analysis.id,
                    "Name": analysis.name,
                    "Kernel Type": analysis.kernel_type,
                    "Date Created": analysis.created_at,
                    "Description": analysis.description
                })
            
            analyses_df = pd.DataFrame(analyses_data)
            
            # Display analyses
            st.dataframe(analyses_df)
            
            # Select analysis to load
            analysis_id = st.selectbox(
                "Select analysis to load:",
                options=[a.id for a in analyses],
                format_func=lambda x: f"ID {x}: {next((a.name for a in analyses if a.id == x), '')}"
            )
            
            # Load selected analysis
            if st.button("Load Selected Analysis"):
                selected_analysis = get_analysis_result_by_id(analysis_id)
                
                if selected_analysis:
                    st.markdown(f"### {selected_analysis.name}")
                    st.markdown(f"**Description:** {selected_analysis.description}")
                    st.markdown(f"**Kernel Type:** {selected_analysis.kernel_type}")
                    st.markdown(f"**Created:** {selected_analysis.created_at}")
                    
                    # Display kernel parameters
                    st.markdown("#### Kernel Parameters:")
                    st.json(selected_analysis.kernel_params)
                    
                    # Extract visualization data
                    vis_data = selected_analysis.visualization_data
                    
                    # Check if we have all required data
                    if all(k in vis_data for k in ["data_type", "X", "y", "X_transformed"]):
                        # Convert lists back to numpy arrays
                        X = np.array(vis_data["X"])
                        y = np.array(vis_data["y"])
                        X_transformed = np.array(vis_data["X_transformed"])
                        data_type = vis_data["data_type"]
                        
                        # Plot the transformed data
                        fig = plot_transformed_data(
                            X, X_transformed, y, 
                            selected_analysis.kernel_type.capitalize(), 
                            data_type.capitalize()
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Visualization data is incomplete or in an unexpected format.")
                else:
                    st.error("Could not load the selected analysis.")

# Initialize Streamlit app
if __name__ == "__main__":
    # Ensure database is initialized
    init_db()
    seed_database()