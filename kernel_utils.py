import numpy as np
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel, sigmoid_kernel
from sklearn.manifold import TSNE

def compute_kernel_matrix(X, kernel_type, **params):
    """
    Compute the kernel matrix for a given dataset and kernel type
    
    Parameters:
    X: numpy array of shape (n_samples, n_features)
    kernel_type: str, type of kernel (Linear, RBF, Polynomial, Sigmoid)
    params: additional parameters for the kernel
    
    Returns:
    K: numpy array of shape (n_samples, n_samples), the kernel matrix
    """
    if kernel_type == "Linear":
        return linear_kernel(X)
    
    elif kernel_type == "RBF":
        gamma = params.get("gamma", 1.0)
        return rbf_kernel(X, gamma=gamma)
    
    elif kernel_type == "Polynomial":
        degree = params.get("degree", 3)
        coef0 = params.get("coef0", 1.0)
        return polynomial_kernel(X, degree=degree, coef0=coef0)
    
    elif kernel_type == "Sigmoid":
        gamma = params.get("gamma", 1.0)
        coef0 = params.get("coef0", 1.0)
        return sigmoid_kernel(X, gamma=gamma, coef0=coef0)
    
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


def compute_krein_kernel_matrix(X, pos_gamma=0.5, neg_weight=0.3):
    """
    Compute a Kreĭn space kernel matrix with both positive and negative components
    
    Parameters:
    X: numpy array of shape (n_samples, n_features)
    pos_gamma: float, gamma parameter for the positive component
    neg_weight: float, weight of the negative component (0-1)
    
    Returns:
    K: numpy array of shape (n_samples, n_samples), the Kreĭn kernel matrix
    """
    # Positive definite component (RBF kernel)
    K_pos = rbf_kernel(X, gamma=pos_gamma)
    
    # Indefinite/negative component (create from data patterns)
    # Simplified for demonstration: using cosine-like negative component
    n_samples = X.shape[0]
    K_neg = np.zeros((n_samples, n_samples))
    
    # Create a pattern of negative similarity (for demonstration)
    for i in range(n_samples):
        for j in range(n_samples):
            # Calculate Euclidean distance
            dist = np.sum((X[i] - X[j])**2)
            # Create oscillating pattern based on distance
            K_neg[i, j] = np.cos(dist * 2) * np.exp(-0.1 * dist)
    
    # Combine both components
    K_krein = K_pos - (neg_weight * K_neg)
    
    return K_krein


def apply_kernel_transformation(X, kernel_type, **params):
    """
    Apply kernel transformation to data for visualization
    
    Parameters:
    X: numpy array of shape (n_samples, n_features)
    kernel_type: str, type of kernel
    params: additional parameters for the kernel
    
    Returns:
    X_transformed: numpy array of shape (n_samples, 2), transformed data for visualization
    """
    # Compute the kernel matrix
    if kernel_type == "Kreĭn-Space":
        pos_gamma = params.get("pos_gamma", 0.5)
        neg_weight = params.get("neg_weight", 0.3)
        K = compute_krein_kernel_matrix(X, pos_gamma, neg_weight)
    else:
        K = compute_kernel_matrix(X, kernel_type, **params)
    
    # Use t-SNE to project the kernel matrix to 2D for visualization
    tsne = TSNE(n_components=2, random_state=42)
    X_transformed = tsne.fit_transform(K)
    
    return X_transformed
