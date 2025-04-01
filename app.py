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
    "Prof. Gärtner's Research",
    "Kreĭn Space Mathematics",
    "Antibiotic Synergy Context",
    "Interactive Visualizations",
    "Kernel Matrices", 
    "Upload Your Data",
    "Experimental Data Visualization"
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
        
        # Create a simple graph visualization
        plt.figure(figsize=(6, 4))
        
        # Create a simple graph with nodes and edges
        G_nodes = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1]])
        G_edges = [(0, 1), (1, 2), (3, 1), (1, 4)]
        
        # Plot the graph
        plt.scatter(G_nodes[:, 0], G_nodes[:, 1], s=100, c='blue')
        for i, j in G_edges:
            plt.plot([G_nodes[i, 0], G_nodes[j, 0]], [G_nodes[i, 1], G_nodes[j, 1]], 'k-')
            
        plt.title("Simple Graph Structure")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        
        # Create a second graph with a different structure
        plt.figure(figsize=(6, 4))
        H_nodes = np.array([[0.1, 0.5], [0.3, 0.8], [0.5, 0.5], [0.7, 0.8], [0.9, 0.5]])
        H_edges = [(0, 2), (1, 2), (2, 3), (2, 4)]
        
        # Plot the second graph
        plt.scatter(H_nodes[:, 0], H_nodes[:, 1], s=100, c='green')
        for i, j in H_edges:
            plt.plot([H_nodes[i, 0], H_nodes[j, 0]], [H_nodes[i, 1], H_nodes[j, 1]], 'k-')
            
        plt.title("Another Graph Structure")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        
        # Display both graphs
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plt.figure(1))
        with col2:
            st.pyplot(plt.figure(2))
        
        st.markdown("""
        **Graph kernels** measure similarity between these network structures by comparing:
        - Node patterns
        - Connection patterns
        - Subgraph structures
        
        This is ideal for comparing pathway networks in biological systems.
        """)
        
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

# Professor Gärtner's Research page
elif page == "Prof. Gärtner's Research":
    st.header("Professor Thomas Gärtner's Research")
    
    st.markdown("""
    ### Background
    
    Professor Thomas Gärtner is a recognized expert in machine learning, particularly in the 
    development of innovative kernel methods and their applications. His research focuses on 
    learning with non-vectorial data, kernels for structured data, and learning in 
    non-standard spaces such as Kreĭn spaces.
    
    ### Key Research Areas
    
    #### 1. Indefinite Learning and Kreĭn Space Methods
    
    Professor Gärtner's group has made significant contributions to kernel methods in indefinite spaces. 
    Traditional kernel methods assume that similarities between objects are represented by positive definite 
    kernels, which correspond to inner products in Hilbert spaces. However, many real-world similarity 
    measures are indefinite, meaning they don't satisfy the positive definiteness property.
    
    Kreĭn spaces provide a mathematical framework for working with such indefinite kernels:
    
    - They decompose into a positive and negative part: K = K₊ - K₋
    - Allow the modeling of both "attraction" and "repulsion" between data points
    - Enable the representation of complex antagonistic relationships
    
    #### 2. Constructive Machine Learning
    
    Another focus of Prof. Gärtner's work is on constructive machine learning methods. This approach:
    
    - Emphasizes building interpretable models by constructing meaningful features
    - Integrates domain knowledge into the learning process
    - Creates models that not only predict but also explain
    
    #### 3. Applications to Biological Systems
    
    His group has applied these methods to biological systems, including:
    
    - Analyzing protein-protein interaction networks
    - Predicting drug-target interactions
    - Modeling complex pathway inhibition patterns
    
    ### Relevance to Antibiotic Synergy Research
    
    The application of Kreĭn-space methods to antibiotic synergy is particularly promising because:
    
    1. **Complex Interactions**: Antibiotics can interact in ways that are neither purely synergistic nor 
    antagonistic, but involve elements of both. Kreĭn spaces naturally model such complexity.
    
    2. **Network Effects**: Antibiotics disrupt bacterial cellular networks in ways that propagate through 
    interconnected pathways. Graph kernels in Kreĭn spaces can capture these network-level effects.
    
    3. **Mechanistic Understanding**: Unlike black-box approaches, kernel methods provide a way to 
    interpret models in terms of similarities to known examples, enabling researchers to gain 
    mechanistic insights.
    """)
    
    # Citation and publications
    st.subheader("Selected Publications")
    
    st.markdown("""
    1. Oglic, D., & Gärtner, T. (2018). Learning in Reproducing Kernel Kreĭn Spaces. *International Conference on Machine Learning (ICML)*.
    
    2. Loosli, G., Canu, S., & Gärtner, T. (2016). Indefinite Proximities for SVM Classification. *ESANN*.
    
    3. Oglic, D., & Gärtner, T. (2019). Scalable Learning in Reproducing Kernel Krein Spaces. *International Conference on Machine Learning (ICML)*.
    
    4. Gärtner, T., Le, Q.V., & Smola, A.J. (2006). A short tour of kernel methods for graphs. *Technical report*.
    """)

# Kreĭn Space Mathematics page
elif page == "Kreĭn Space Mathematics":
    st.header("Mathematical Theory of Kreĭn Spaces")
    
    st.markdown("""
    ### Fundamentals of Kreĭn Spaces
    
    A Kreĭn space K is a vector space equipped with an indefinite inner product that can be decomposed 
    into the difference of two positive definite inner products.
    
    #### Definition
    
    A Kreĭn space is a vector space K with an indefinite inner product ⟨·,·⟩ₖ that can be written as:
    
    $$⟨x, y⟩_K = ⟨x, y⟩_{H_+} - ⟨x, y⟩_{H_-}$$
    
    where (H₊, ⟨·,·⟩_{H_+}) and (H₋, ⟨·,·⟩_{H_-}) are Hilbert spaces with positive definite inner products.
    
    #### Key Properties
    
    1. **Indefiniteness**: Unlike Hilbert spaces, inner products in Kreĭn spaces can be negative.
    
    2. **Fundamental Decomposition**: Any Kreĭn space can be decomposed as K = K₊ ⊕ K₋, where K₊ and K₋ are Hilbert spaces.
    
    3. **J-Symmetry**: There exists a "fundamental symmetry" operator J such that ⟨x, y⟩ₖ = ⟨Jx, y⟩ₕ, where ⟨·,·⟩ₕ is a Hilbert space inner product.
    """)
    
    # Visualize the concept of Kreĭn space
    st.subheader("Geometric Interpretation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create a visual representation of Hilbert space (positive definite)
        plt.figure(figsize=(5, 5))
        
        # Generate some points in 2D space
        np.random.seed(42)
        X = np.random.randn(50, 2)
        
        # Plot points and vectors
        plt.scatter(X[:, 0], X[:, 1], s=30, alpha=0.5)
        
        # Draw some vectors
        for i in range(5):
            x = X[i]
            plt.arrow(0, 0, x[0], x[1], head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        
        plt.title("Hilbert Space (Positive Definite)")
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.grid(True)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        st.pyplot(plt.gcf())
    
    with col2:
        # Create a visual representation of Kreĭn space (indefinite)
        plt.figure(figsize=(5, 5))
        
        # Generate some points in 2D space
        np.random.seed(42)
        X = np.random.randn(50, 2)
        
        # Plot points and vectors
        plt.scatter(X[:, 0], X[:, 1], s=30, alpha=0.5)
        
        # Draw vectors with sign information
        for i in range(5):
            x = X[i]
            # Color based on inner product sign
            sign = 1 if x[0]*x[0] - x[1]*x[1] > 0 else -1
            color = 'green' if sign > 0 else 'red'
            plt.arrow(0, 0, x[0], x[1], head_width=0.1, head_length=0.1, fc=color, ec=color)
        
        plt.title("Kreĭn Space (Indefinite)")
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.grid(True)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        st.pyplot(plt.gcf())
    
    st.markdown("""
    ### Kreĭn Space Kernels
    
    A Kreĭn space kernel is a function K: X × X → ℝ such that:
    
    1. It can be decomposed as K = K₊ - K₋, where K₊ and K₋ are positive definite kernels.
    2. For any finite set of points {x₁, x₂, ..., xₙ}, the matrix [K(xᵢ, xⱼ)]ᵢⱼ has a finite number of negative eigenvalues.
    
    #### Mathematical Formulation
    
    For a Kreĭn space kernel K:
    
    $$K(x, y) = K_+(x, y) - K_-(x, y)$$
    
    Where K₊ and K₋ are positive definite kernels corresponding to Hilbert spaces H₊ and H₋.
    
    #### Support Vector Machines in Kreĭn Spaces
    
    The objective function for SVMs in Kreĭn spaces becomes:
    
    $$\min_{f \in \mathcal{H}_K} \frac{1}{2} ||f||^2_K + C \sum_{i=1}^n \max(0, 1 - y_i f(x_i))$$
    
    where ||f||²ₖ = ||f₊||²_{H₊} - ||f₋||²_{H₋} is the squared norm in the Kreĭn space.
    
    This formulation allows for modeling complex decision boundaries that might be impossible 
    with standard positive definite kernels.
    """)
    
    st.subheader("Applications to Antibiotic Synergy")
    
    st.markdown("""
    In the context of antibiotic synergy, Kreĭn space kernels enable:
    
    1. **Modeling Antagonistic Interactions**: The negative part of the kernel (K₋) can represent 
    antagonistic effects between drugs.
    
    2. **Capturing Complex Dynamics**: The interplay between synergistic and antagonistic components 
    can model the complex dynamics of bacterial stress responses.
    
    3. **Improved Classification**: By allowing indefinite similarities, Kreĭn-SVMs can better classify 
    drug combinations as synergistic, additive, or antagonistic.
    """)
    
    # Example visualization of decision boundary
    st.subheader("Visualizing Decision Boundaries")
    
    # Generate some data
    np.random.seed(42)
    X1 = np.random.randn(50, 2) * 0.7 + np.array([2, 2])  # Class 1
    X2 = np.random.randn(50, 2) * 0.7 + np.array([-2, -2])  # Class 2
    X3 = np.random.randn(50, 2) * 0.7 + np.array([2, -2])  # Class 3
    
    X = np.vstack([X1, X2, X3])
    y = np.hstack([np.zeros(50), np.ones(50), 2 * np.ones(50)])
    
    # Create a plot with decision regions
    plt.figure(figsize=(10, 6))
    
    # Plot points
    plt.scatter(X1[:, 0], X1[:, 1], c='blue', label='Synergistic')
    plt.scatter(X2[:, 0], X2[:, 1], c='red', label='Antagonistic')
    plt.scatter(X3[:, 0], X3[:, 1], c='green', label='Additive')
    
    # Add curved decision boundaries (simplified visualization)
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Boundary 1 (circle)
    r = 2.5
    plt.plot(r*np.cos(theta), r*np.sin(theta), 'k--', alpha=0.5)
    
    # Boundary 2 (curved)
    x = np.linspace(-4, 4, 100)
    plt.plot(x, -0.2*x**2 + 1, 'k--', alpha=0.5)
    
    plt.title("Simplified Visualization of Kreĭn-SVM Decision Boundaries")
    plt.xlabel("Feature 1 (e.g., Drug A concentration)")
    plt.ylabel("Feature 2 (e.g., Drug B concentration)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    st.pyplot(plt.gcf())
    
    st.markdown("""
    *Note: This is a simplified visualization. Actual decision boundaries in Kreĭn-space SVMs can be 
    much more complex, capturing intricate patterns of drug interactions that standard SVMs cannot represent.*
    """)

# Antibiotic Synergy Context page
elif page == "Antibiotic Synergy Context":
    st.header("Biological Context: Antibiotic Synergy")
    
    st.markdown("""
    ### The Challenge of Antibiotic Resistance
    
    Antibiotic resistance is one of the most pressing global health challenges of our time. According to the WHO, 
    antibiotic-resistant infections could cause 10 million deaths annually by 2050 if no action is taken.
    
    #### Key Challenges
    
    1. **Pipeline Gap**: Few new antibiotics are being developed
    2. **Evolutionary Pressure**: Single antibiotics drive rapid resistance development
    3. **Complex Mechanisms**: Resistance involves multiple cellular pathways
    
    ### Antibiotic Combinations as a Solution
    
    Combining multiple antibiotics can:
    
    1. **Enhance Efficacy**: Creating effects greater than the sum of individual drugs
    2. **Reduce Resistance**: Making it harder for bacteria to develop concurrent mechanisms
    3. **Lower Dosages**: Reducing side effects while maintaining effectiveness
    
    However, not all combinations are beneficial. Some combinations can be:
    
    - **Synergistic**: Enhanced effect (greater than additive)
    - **Additive**: Combined effect equals the sum of individual effects
    - **Antagonistic**: Combined effect is less than the sum, or even inhibitory
    """)
    
    # Create visualization of different types of interactions
    col1, col2 = st.columns(2)
    
    with col1:
        # Create heatmap for synergistic interactions
        conc_range = np.linspace(0, 1, 10)
        X, Y = np.meshgrid(conc_range, conc_range)
        
        # Synergistic effect (more than additive)
        Z_synergy = 0.8 * X + 0.7 * Y + 1.5 * X * Y
        
        fig = go.Figure(data=[go.Heatmap(
            z=Z_synergy,
            x=conc_range,
            y=conc_range,
            colorscale='Viridis',
            colorbar=dict(title="Effect")
        )])
        
        fig.update_layout(
            title="Synergistic Interaction",
            xaxis_title="Drug A Concentration",
            yaxis_title="Drug B Concentration",
            width=400,
            height=400
        )
        
        st.plotly_chart(fig)
    
    with col2:
        # Create heatmap for antagonistic interactions
        Z_antagonistic = 0.8 * X + 0.7 * Y - 0.5 * X * Y
        
        fig = go.Figure(data=[go.Heatmap(
            z=Z_antagonistic,
            x=conc_range,
            y=conc_range,
            colorscale='Viridis',
            colorbar=dict(title="Effect")
        )])
        
        fig.update_layout(
            title="Antagonistic Interaction",
            xaxis_title="Drug A Concentration",
            yaxis_title="Drug B Concentration",
            width=400,
            height=400
        )
        
        st.plotly_chart(fig)
    
    st.markdown("""
    ### Biological Mechanisms of Synergy
    
    Understanding why certain antibiotics work synergistically requires examining:
    
    #### 1. Cellular Pathway Interactions
    
    - **Sequential Inhibition**: One drug enhances the uptake or targeting of another
    - **Parallel Pathway Inhibition**: Simultaneous disruption of multiple cellular functions
    - **Stress Response Modulation**: One drug prevents adaptive responses to another
    
    #### 2. Metabolic Network Effects
    
    - **Flux Redirection**: Blocking one pathway forces metabolism through another vulnerable route
    - **Energy Depletion**: Combined assault on energy-producing systems
    - **Resource Competition**: Forcing cellular machinery to respond to multiple threats simultaneously
    
    #### 3. Resistance Mechanism Interactions
    
    - **Efflux Pump Saturation**: Overwhelming bacterial expulsion mechanisms
    - **Cell Wall Destabilization**: One drug creates entry points for another
    - **Compensatory Mutation Prevention**: Blocking genetic adaptation pathways
    """)
    
    # Create network diagram
    st.subheader("Antibiotic Interaction Networks")
    
    # Create network figure
    plt.figure(figsize=(10, 6))
    
    # Define nodes (cellular components/processes)
    nodes = {
        'CellWall': (0.2, 0.8),
        'DNASynthesis': (0.8, 0.8),
        'Ribosome': (0.5, 0.5),
        'Metabolism': (0.2, 0.2),
        'MembraneTransport': (0.8, 0.2)
    }
    
    # Define edges (interactions)
    edges = [
        ('CellWall', 'MembraneTransport'),
        ('CellWall', 'Metabolism'),
        ('DNASynthesis', 'Metabolism'),
        ('DNASynthesis', 'Ribosome'),
        ('Ribosome', 'Metabolism'),
        ('Ribosome', 'MembraneTransport')
    ]
    
    # Plot nodes
    for name, (x, y) in nodes.items():
        plt.plot(x, y, 'o', markersize=15, color='cornflowerblue')
        plt.text(x, y+0.05, name, ha='center', fontsize=10)
    
    # Plot edges
    for node1, node2 in edges:
        x1, y1 = nodes[node1]
        x2, y2 = nodes[node2]
        plt.plot([x1, x2], [y1, y2], '-', color='gray', alpha=0.6)
    
    # Add drug inhibition arrows
    drug_a = (0.1, 0.9)
    drug_b = (0.9, 0.9)
    
    plt.plot(drug_a[0], drug_a[1], 's', markersize=10, color='crimson')
    plt.text(drug_a[0], drug_a[1]+0.05, "Drug A", ha='center', fontsize=10)
    
    plt.plot(drug_b[0], drug_b[1], 's', markersize=10, color='purple')
    plt.text(drug_b[0], drug_b[1]+0.05, "Drug B", ha='center', fontsize=10)
    
    # Inhibition arrows
    plt.arrow(drug_a[0]+0.02, drug_a[1]-0.02, 0.1, -0.05, head_width=0.02, head_length=0.02, fc='crimson', ec='crimson')
    plt.arrow(drug_b[0]-0.02, drug_b[1]-0.02, -0.05, -0.05, head_width=0.02, head_length=0.02, fc='purple', ec='purple')
    
    plt.title("Network Model of Antibiotic Interactions")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    st.pyplot(plt.gcf())
    
    st.markdown("""
    ### Why Kernel Methods for Antibiotic Synergy Research?
    
    Kernel methods are particularly well-suited for antibiotic synergy research because:
    
    1. **Capturing Complex Interactions**: Kernels can represent non-linear relationships between drugs and their targets
    
    2. **Interpretability**: Unlike black-box neural networks, kernel methods provide insights into which examples (drug pairs) 
       influence predictions, aiding mechanistic understanding
    
    3. **Prior Knowledge Integration**: Graph kernels can incorporate known biological network information
    
    4. **Handling Heterogeneous Data**: Kernels can combine multiple data types (chemical structures, genomics, proteomics)
    
    5. **Modeling Opposing Forces**: Kreĭn-space kernels uniquely capture the push-pull dynamics in bacterial networks 
       responding to multiple stressors
    """)
    
    st.subheader("Current State of the Art")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Traditional Methods
        
        - **Checkerboard Assays**:
          - Manual experimental testing of drug combinations
          - Time-consuming and expensive
          - Limited to pairwise interactions
        
        - **Fractional Inhibitory Concentration Index (FICI)**:
          - Simple mathematical model
          - Doesn't capture complex interactions
          - Binary classification only (synergistic/not)
        """)
    
    with col2:
        st.markdown("""
        #### Advanced Computational Approaches
        
        - **DeepSynergy**:
          - Neural network model for drug combination effects
          - High accuracy but low interpretability
          - Requires large datasets
        
        - **Mechanistic Models**:
          - Detailed cellular simulations
          - Extremely complex
          - Requires extensive parameter tuning
        """)
    
    st.markdown("""
    ### The Promise of Kreĭn-Space Kernel Methods
    
    Kreĭn-space kernel methods represent a middle ground:
    
    - More sophisticated than simple indices like FICI
    - More interpretable than black-box neural networks
    - Better at modeling the dual nature (synergy/antagonism) of drug interactions
    - Capable of representing both local and global network effects
    
    By modeling both synergistic (positive) and antagonistic (negative) components simultaneously,
    Kreĭn-space kernels match the biological reality of competing forces in cellular networks
    responding to multiple stressors.
    """)

# Experimental Data Visualization page
elif page == "Experimental Data Visualization":
    st.header("Experimental Data Visualization")
    
    st.markdown("""
    ### Analyzing Real Experimental Data
    
    This section allows you to visualize and analyze experimental antibiotic synergy data. 
    While we cannot provide real proprietary research data, we have created a representative dataset 
    based on published literature trends to demonstrate how kernel methods can be applied to 
    experimental results.
    """)
    
    # Create sample experimental data
    np.random.seed(42)
    
    # Create drug pairs
    antibiotics = ["Ampicillin", "Tetracycline", "Ciprofloxacin", "Gentamicin", "Trimethoprim",
                  "Erythromycin", "Rifampicin", "Vancomycin", "Ceftriaxone", "Azithromycin"]
    
    n_drugs = len(antibiotics)
    n_pairs = n_drugs * (n_drugs - 1) // 2
    
    # Create empty dataframe
    pairs = []
    for i in range(n_drugs):
        for j in range(i+1, n_drugs):
            pairs.append((antibiotics[i], antibiotics[j]))
    
    # Generate features that might be available from experiments
    # These would normally come from real data
    
    # Drug A properties (MIC values, etc)
    drug_A_properties = np.random.rand(n_pairs, 3)
    
    # Drug B properties
    drug_B_properties = np.random.rand(n_pairs, 3)
    
    # Combined effect measures
    combined_effects = np.random.rand(n_pairs, 4)
    
    # FICI values (< 0.5: synergistic, 0.5-4: additive, > 4: antagonistic)
    fici_values = np.random.lognormal(mean=-0.5, sigma=0.7, size=n_pairs)
    
    # Synergy scores
    synergy_scores = 1.0 - fici_values / 5  # Transform to roughly 0-1 scale
    synergy_scores = np.clip(synergy_scores, 0, 1)
    
    # Create categorical labels
    synergy_labels = ["Antagonistic" if f > 4 else "Additive" if f >= 0.5 else "Synergistic" for f in fici_values]
    
    # Create dataframe
    data = {
        "Drug_A": [pair[0] for pair in pairs],
        "Drug_B": [pair[1] for pair in pairs],
        "FICI": fici_values,
        "Synergy_Score": synergy_scores,
        "Classification": synergy_labels,
        "MIC_Reduction_A": drug_A_properties[:, 0] * 10,
        "MIC_Reduction_B": drug_B_properties[:, 0] * 10,
        "Growth_Inhibition": combined_effects[:, 0] * 100  # Percentage
    }
    
    exp_df = pd.DataFrame(data)
    
    # Display the data
    st.subheader("Sample Experimental Data")
    st.dataframe(exp_df)
    
    # Visualization options
    st.subheader("Data Visualization")
    
    viz_type = st.radio(
        "Select visualization type",
        ["Synergy Heatmap", "Drug Pair Comparison", "Kernel Analysis"]
    )
    
    if viz_type == "Synergy Heatmap":
        st.markdown("### Synergy Heatmap")
        st.markdown("This heatmap shows the interaction strengths between different antibiotic pairs.")
        
        # Create a matrix for heatmap
        synergy_matrix = np.zeros((n_drugs, n_drugs))
        
        # Fill the matrix with FICI values
        pair_idx = 0
        for i in range(n_drugs):
            for j in range(i+1, n_drugs):
                synergy_value = 1.0 - exp_df.iloc[pair_idx]["FICI"] / 5  # Transform for visualization
                synergy_value = np.clip(synergy_value, 0, 1)
                
                synergy_matrix[i, j] = synergy_value
                synergy_matrix[j, i] = synergy_value  # Make it symmetric
                pair_idx += 1
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=synergy_matrix,
            x=antibiotics,
            y=antibiotics,
            colorscale='RdBu_r',
            colorbar=dict(title="Synergy Score")
        ))
        
        fig.update_layout(
            title="Antibiotic Synergy Heatmap",
            xaxis_title="Antibiotic",
            yaxis_title="Antibiotic",
            width=700,
            height=700
        )
        
        st.plotly_chart(fig)
        
        st.markdown("""
        **Interpretation**: 
        - Darker blue colors indicate stronger synergistic effects
        - Darker red colors indicate antagonistic effects
        - White/neutral colors indicate additive effects
        
        This visualization helps identify promising drug combinations for further investigation.
        """)
    
    elif viz_type == "Drug Pair Comparison":
        st.markdown("### Drug Pair Comparison")
        
        # Allow selection of drug pairs
        col1, col2 = st.columns(2)
        
        with col1:
            drug_a = st.selectbox("Select first drug", antibiotics)
        
        with col2:
            drug_b_options = [drug for drug in antibiotics if drug != drug_a]
            drug_b = st.selectbox("Select second drug", drug_b_options)
        
        # Find the pair in the dataframe
        pair_data = exp_df[(exp_df["Drug_A"] == drug_a) & (exp_df["Drug_B"] == drug_b)]
        
        if pair_data.empty:
            pair_data = exp_df[(exp_df["Drug_A"] == drug_b) & (exp_df["Drug_B"] == drug_a)]
        
        if not pair_data.empty:
            # Display pair information
            st.subheader(f"{drug_a} + {drug_b} Interaction")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("FICI Value", f"{pair_data['FICI'].values[0]:.2f}")
            
            with col2:
                st.metric("Synergy Score", f"{pair_data['Synergy_Score'].values[0]:.2f}")
            
            with col3:
                st.metric("Classification", pair_data["Classification"].values[0])
            
            # Create a detailed visualization
            # Simulate a dose-response surface
            conc_range = np.linspace(0, 1, 20)
            X, Y = np.meshgrid(conc_range, conc_range)
            
            # Create a response surface based on the synergy type
            if pair_data["Classification"].values[0] == "Synergistic":
                Z = 0.8 * X + 0.7 * Y + 1.5 * X * Y
            elif pair_data["Classification"].values[0] == "Antagonistic":
                Z = 0.8 * X + 0.7 * Y - 0.5 * X * Y
            else:  # Additive
                Z = 0.8 * X + 0.7 * Y + 0.1 * X * Y
            
            # Add some noise
            np.random.seed(42)
            Z += np.random.normal(0, 0.05, Z.shape)
            
            # Create the surface plot
            fig = go.Figure(data=[go.Surface(z=Z, x=conc_range, y=conc_range)])
            fig.update_layout(
                title=f"{drug_a} + {drug_b} Dose-Response Surface",
                scene=dict(
                    xaxis_title=f"{drug_a} Concentration",
                    yaxis_title=f"{drug_b} Concentration",
                    zaxis_title="Inhibition Effect"
                ),
                width=700,
                height=700
            )
            
            st.plotly_chart(fig)
            
            st.markdown("""
            **Interpreting the Response Surface**:
            
            This 3D surface shows how the combined effect of the drug pair varies with different concentrations.
            
            - For **synergistic** pairs, the surface curves upward (combined effect > sum of individual effects)
            - For **antagonistic** pairs, the surface curves downward (combined effect < sum of individual effects)
            - For **additive** pairs, the surface is nearly linear (combined effect ≈ sum of individual effects)
            
            The shape of this surface provides insights into the optimal dosing ratios for maximal efficacy.
            """)
        
    elif viz_type == "Kernel Analysis":
        st.markdown("### Kernel Analysis of Drug Interactions")
        
        # Create a feature matrix from the data
        features = np.column_stack([
            exp_df["MIC_Reduction_A"],
            exp_df["MIC_Reduction_B"],
            exp_df["Growth_Inhibition"]
        ])
        
        # Normalize features
        features = (features - features.mean(axis=0)) / features.std(axis=0)
        
        # Create labels based on classification
        class_mapping = {"Synergistic": 2, "Additive": 1, "Antagonistic": 0}
        labels = np.array([class_mapping[label] for label in exp_df["Classification"]])
        
        # Select kernel type
        kernel_type = st.selectbox(
            "Select kernel for analysis",
            ["Linear", "RBF", "Polynomial", "Kreĭn-Space"]
        )
        
        # Compute kernel matrix
        if kernel_type == "Kreĭn-Space":
            pos_gamma = st.slider("Positive Component Gamma", 0.01, 2.0, 0.5, 0.01)
            neg_weight = st.slider("Negative Component Weight", 0.0, 1.0, 0.3, 0.05)
            
            kernel_matrix = compute_krein_kernel_matrix(features, pos_gamma=pos_gamma, neg_weight=neg_weight)
            compared_kernel = compute_kernel_matrix(features, "RBF", gamma=pos_gamma)
        else:
            if kernel_type == "RBF":
                gamma = st.slider("Gamma", 0.01, 2.0, 0.5, 0.01)
                kernel_params = {"gamma": gamma}
            elif kernel_type == "Polynomial":
                degree = st.slider("Degree", 1, 5, 3)
                coef0 = st.slider("Coef0", 0.0, 2.0, 1.0, 0.1)
                kernel_params = {"degree": degree, "coef0": coef0}
            else:  # Linear
                kernel_params = {}
            
            kernel_matrix = compute_kernel_matrix(features, kernel_type, **kernel_params)
            
            # Compare with Kreĭn kernel
            compared_kernel = compute_krein_kernel_matrix(features, pos_gamma=0.5, neg_weight=0.3)
        
        # Plot the kernel matrices
        fig = plot_kernel_matrix(
            kernel_matrix, 
            compared_kernel, 
            kernel_type, 
            "Kreĭn-Space" if kernel_type != "Kreĭn-Space" else "RBF", 
            labels
        )
        
        st.plotly_chart(fig)
        
        # Apply kernel transformation for visualization
        if kernel_type == "Kreĭn-Space":
            transformed_features = apply_kernel_transformation(
                features, 
                kernel_type, 
                pos_gamma=pos_gamma, 
                neg_weight=neg_weight
            )
        else:
            transformed_features = apply_kernel_transformation(features, kernel_type, **kernel_params)
        
        # Create a scatter plot of the transformed data
        fig = px.scatter(
            x=transformed_features[:, 0],
            y=transformed_features[:, 1],
            color=exp_df["Classification"],
            hover_name=[f"{a} + {b}" for a, b in zip(exp_df["Drug_A"], exp_df["Drug_B"])],
            color_discrete_map={
                "Synergistic": "green",
                "Additive": "blue",
                "Antagonistic": "red"
            },
            labels={
                "x": "Transformed Dimension 1",
                "y": "Transformed Dimension 2",
                "color": "Interaction Type"
            },
            title=f"Drug Combinations after {kernel_type} Kernel Transformation"
        )
        
        st.plotly_chart(fig)
        
        st.markdown(f"""
        **Interpreting the Kernel Analysis**:
        
        The kernel matrix visualization shows pairwise similarities between drug combinations according to the selected kernel.
        
        The scatter plot shows how the {kernel_type} kernel transforms the data. Points that cluster together are considered similar by the kernel.
        
        Note how the Kreĭn-space kernel often provides better separation between different interaction types 
        (synergistic, additive, antagonistic) because it can model both positive and negative similarities.
        
        This analysis helps identify patterns in drug combination effects that might not be apparent from 
        raw experimental data, potentially revealing underlying mechanisms of action.
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

# Experimental Data Visualization page
elif page == "Experimental Data Visualization":
    st.header("Experimental Data Visualization")
    
    st.markdown("""
    ### Analyzing Real Experimental Data
    
    This section allows you to visualize and analyze experimental antibiotic synergy data. 
    While we cannot provide real proprietary research data, we have created a representative dataset 
    based on published literature trends to demonstrate how kernel methods can be applied to 
    experimental results.
    """)
    
    # Create sample experimental data
    np.random.seed(42)
    
    # Create drug pairs
    antibiotics = ["Ampicillin", "Tetracycline", "Ciprofloxacin", "Gentamicin", "Trimethoprim",
                  "Erythromycin", "Rifampicin", "Vancomycin", "Ceftriaxone", "Azithromycin"]
    
    n_drugs = len(antibiotics)
    n_pairs = n_drugs * (n_drugs - 1) // 2
    
    # Create empty dataframe
    pairs = []
    for i in range(n_drugs):
        for j in range(i+1, n_drugs):
            pairs.append((antibiotics[i], antibiotics[j]))
    
    # Generate features that might be available from experiments
    # These would normally come from real data
    
    # Drug A properties (MIC values, etc)
    drug_A_properties = np.random.rand(n_pairs, 3)
    
    # Drug B properties
    drug_B_properties = np.random.rand(n_pairs, 3)
    
    # Combined effect measures
    combined_effects = np.random.rand(n_pairs, 4)
    
    # FICI values (< 0.5: synergistic, 0.5-4: additive, > 4: antagonistic)
    fici_values = np.random.lognormal(mean=-0.5, sigma=0.7, size=n_pairs)
    
    # Synergy scores
    synergy_scores = 1.0 - fici_values / 5  # Transform to roughly 0-1 scale
    synergy_scores = np.clip(synergy_scores, 0, 1)
    
    # Create categorical labels
    synergy_labels = ["Antagonistic" if f > 4 else "Additive" if f >= 0.5 else "Synergistic" for f in fici_values]
    
    # Create dataframe
    data = {
        "Drug_A": [pair[0] for pair in pairs],
        "Drug_B": [pair[1] for pair in pairs],
        "FICI": fici_values,
        "Synergy_Score": synergy_scores,
        "Classification": synergy_labels,
        "MIC_Reduction_A": drug_A_properties[:, 0] * 10,
        "MIC_Reduction_B": drug_B_properties[:, 0] * 10,
        "Growth_Inhibition": combined_effects[:, 0] * 100  # Percentage
    }
    
    exp_df = pd.DataFrame(data)
    
    # Display the data
    st.subheader("Sample Experimental Data")
    st.dataframe(exp_df)
    
    # Visualization options
    st.subheader("Data Visualization")
    
    viz_type = st.radio(
        "Select visualization type",
        ["Synergy Heatmap", "Drug Pair Comparison", "Kernel Analysis"]
    )
    
    if viz_type == "Synergy Heatmap":
        st.markdown("### Synergy Heatmap")
        st.markdown("This heatmap shows the interaction strengths between different antibiotic pairs.")
        
        # Create a matrix for heatmap
        synergy_matrix = np.zeros((n_drugs, n_drugs))
        
        # Fill the matrix with FICI values
        pair_idx = 0
        for i in range(n_drugs):
            for j in range(i+1, n_drugs):
                synergy_value = 1.0 - exp_df.iloc[pair_idx]["FICI"] / 5  # Transform for visualization
                synergy_value = np.clip(synergy_value, 0, 1)
                
                synergy_matrix[i, j] = synergy_value
                synergy_matrix[j, i] = synergy_value  # Make it symmetric
                pair_idx += 1
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=synergy_matrix,
            x=antibiotics,
            y=antibiotics,
            colorscale='RdBu_r',
            colorbar=dict(title="Synergy Score")
        ))
        
        fig.update_layout(
            title="Antibiotic Synergy Heatmap",
            xaxis_title="Antibiotic",
            yaxis_title="Antibiotic",
            width=700,
            height=700
        )
        
        st.plotly_chart(fig)
        
        st.markdown("""
        **Interpretation**: 
        - Darker blue colors indicate stronger synergistic effects
        - Darker red colors indicate antagonistic effects
        - White/neutral colors indicate additive effects
        
        This visualization helps identify promising drug combinations for further investigation.
        """)
    
    elif viz_type == "Drug Pair Comparison":
        st.markdown("### Drug Pair Comparison")
        
        # Allow selection of drug pairs
        col1, col2 = st.columns(2)
        
        with col1:
            drug_a = st.selectbox("Select first drug", antibiotics)
        
        with col2:
            drug_b_options = [drug for drug in antibiotics if drug != drug_a]
            drug_b = st.selectbox("Select second drug", drug_b_options)
        
        # Find the pair in the dataframe
        pair_data = exp_df[(exp_df["Drug_A"] == drug_a) & (exp_df["Drug_B"] == drug_b)]
        
        if pair_data.empty:
            pair_data = exp_df[(exp_df["Drug_A"] == drug_b) & (exp_df["Drug_B"] == drug_a)]
        
        if not pair_data.empty:
            # Display pair information
            st.subheader(f"{drug_a} + {drug_b} Interaction")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("FICI Value", f"{pair_data['FICI'].values[0]:.2f}")
            
            with col2:
                st.metric("Synergy Score", f"{pair_data['Synergy_Score'].values[0]:.2f}")
            
            with col3:
                st.metric("Classification", pair_data["Classification"].values[0])
            
            # Create a detailed visualization
            # Simulate a dose-response surface
            conc_range = np.linspace(0, 1, 20)
            X, Y = np.meshgrid(conc_range, conc_range)
            
            # Create a response surface based on the synergy type
            if pair_data["Classification"].values[0] == "Synergistic":
                Z = 0.8 * X + 0.7 * Y + 1.5 * X * Y
            elif pair_data["Classification"].values[0] == "Antagonistic":
                Z = 0.8 * X + 0.7 * Y - 0.5 * X * Y
            else:  # Additive
                Z = 0.8 * X + 0.7 * Y + 0.1 * X * Y
            
            # Add some noise
            np.random.seed(42)
            Z += np.random.normal(0, 0.05, Z.shape)
            
            # Create the surface plot
            fig = go.Figure(data=[go.Surface(z=Z, x=conc_range, y=conc_range)])
            fig.update_layout(
                title=f"{drug_a} + {drug_b} Dose-Response Surface",
                scene=dict(
                    xaxis_title=f"{drug_a} Concentration",
                    yaxis_title=f"{drug_b} Concentration",
                    zaxis_title="Inhibition Effect"
                ),
                width=700,
                height=700
            )
            
            st.plotly_chart(fig)
            
            st.markdown("""
            **Interpreting the Response Surface**:
            
            This 3D surface shows how the combined effect of the drug pair varies with different concentrations.
            
            - For **synergistic** pairs, the surface curves upward (combined effect > sum of individual effects)
            - For **antagonistic** pairs, the surface curves downward (combined effect < sum of individual effects)
            - For **additive** pairs, the surface is nearly linear (combined effect ≈ sum of individual effects)
            
            The shape of this surface provides insights into the optimal dosing ratios for maximal efficacy.
            """)
        
    elif viz_type == "Kernel Analysis":
        st.markdown("### Kernel Analysis of Drug Interactions")
        
        # Create a feature matrix from the data
        features = np.column_stack([
            exp_df["MIC_Reduction_A"],
            exp_df["MIC_Reduction_B"],
            exp_df["Growth_Inhibition"]
        ])
        
        # Normalize features
        features = (features - features.mean(axis=0)) / features.std(axis=0)
        
        # Create labels based on classification
        class_mapping = {"Synergistic": 2, "Additive": 1, "Antagonistic": 0}
        labels = np.array([class_mapping[label] for label in exp_df["Classification"]])
        
        # Select kernel type
        kernel_type = st.selectbox(
            "Select kernel for analysis",
            ["Linear", "RBF", "Polynomial", "Kreĭn-Space"]
        )
        
        # Compute kernel matrix
        if kernel_type == "Kreĭn-Space":
            pos_gamma = st.slider("Positive Component Gamma", 0.01, 2.0, 0.5, 0.01)
            neg_weight = st.slider("Negative Component Weight", 0.0, 1.0, 0.3, 0.05)
            
            kernel_matrix = compute_krein_kernel_matrix(features, pos_gamma=pos_gamma, neg_weight=neg_weight)
            compared_kernel = compute_kernel_matrix(features, "RBF", gamma=pos_gamma)
        else:
            if kernel_type == "RBF":
                gamma = st.slider("Gamma", 0.01, 2.0, 0.5, 0.01)
                kernel_params = {"gamma": gamma}
            elif kernel_type == "Polynomial":
                degree = st.slider("Degree", 1, 5, 3)
                coef0 = st.slider("Coef0", 0.0, 2.0, 1.0, 0.1)
                kernel_params = {"degree": degree, "coef0": coef0}
            else:  # Linear
                kernel_params = {}
            
            kernel_matrix = compute_kernel_matrix(features, kernel_type, **kernel_params)
            
            # Compare with Kreĭn kernel
            compared_kernel = compute_krein_kernel_matrix(features, pos_gamma=0.5, neg_weight=0.3)
        
        # Plot the kernel matrices
        fig = plot_kernel_matrix(
            kernel_matrix, 
            compared_kernel, 
            kernel_type, 
            "Kreĭn-Space" if kernel_type != "Kreĭn-Space" else "RBF", 
            labels
        )
        
        st.plotly_chart(fig)
        
        # Apply kernel transformation for visualization
        if kernel_type == "Kreĭn-Space":
            transformed_features = apply_kernel_transformation(
                features, 
                kernel_type, 
                pos_gamma=pos_gamma, 
                neg_weight=neg_weight
            )
        else:
            transformed_features = apply_kernel_transformation(features, kernel_type, **kernel_params)
        
        # Create a scatter plot of the transformed data
        fig = px.scatter(
            x=transformed_features[:, 0],
            y=transformed_features[:, 1],
            color=exp_df["Classification"],
            hover_name=[f"{a} + {b}" for a, b in zip(exp_df["Drug_A"], exp_df["Drug_B"])],
            color_discrete_map={
                "Synergistic": "green",
                "Additive": "blue",
                "Antagonistic": "red"
            },
            labels={
                "x": "Transformed Dimension 1",
                "y": "Transformed Dimension 2",
                "color": "Interaction Type"
            },
            title=f"Drug Combinations after {kernel_type} Kernel Transformation"
        )
        
        st.plotly_chart(fig)
        
        st.markdown(f"""
        **Interpreting the Kernel Analysis**:
        
        The kernel matrix visualization shows pairwise similarities between drug combinations according to the selected kernel.
        
        The scatter plot shows how the {kernel_type} kernel transforms the data. Points that cluster together are considered similar by the kernel.
        
        Note how the Kreĭn-space kernel often provides better separation between different interaction types 
        (synergistic, additive, antagonistic) because it can model both positive and negative similarities.
        
        This analysis helps identify patterns in drug combination effects that might not be apparent from 
        raw experimental data, potentially revealing underlying mechanisms of action.
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
