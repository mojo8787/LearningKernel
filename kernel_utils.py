import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel, sigmoid_kernel
from sklearn.decomposition import KernelPCA
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
    if kernel_type.lower() == 'linear':
        return linear_kernel(X)
    
    elif kernel_type.lower() == 'rbf':
        gamma = params.get('gamma', 1.0)
        return rbf_kernel(X, gamma=gamma)
    
    elif kernel_type.lower() == 'polynomial':
        degree = params.get('degree', 3)
        gamma = params.get('gamma', 1.0)
        coef0 = params.get('coef0', 1.0)
        return polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    
    elif kernel_type.lower() == 'sigmoid':
        gamma = params.get('gamma', 1.0)
        coef0 = params.get('coef0', 1.0)
        return sigmoid_kernel(X, gamma=gamma, coef0=coef0)
    
    elif kernel_type.lower() == 'kreĭn' or kernel_type.lower() == 'krein':
        pos_gamma = params.get('pos_gamma', 0.5)
        neg_weight = params.get('neg_weight', 0.3)
        return compute_krein_kernel_matrix(X, pos_gamma, neg_weight)
    
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")

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
    # Compute positive definite component (RBF kernel)
    K_pos = rbf_kernel(X, gamma=pos_gamma)
    
    # Compute negative definite component
    # For simplicity, we use a distance-based negative component
    n_samples = X.shape[0]
    K_neg = np.zeros((n_samples, n_samples))
    
    # Compute squared Euclidean distances
    for i in range(n_samples):
        for j in range(n_samples):
            # We use the negative of the distance as our negative component
            K_neg[i, j] = -np.sum((X[i] - X[j]) ** 2)
    
    # Normalize K_neg to have similar scale to K_pos
    K_neg = K_neg / np.abs(K_neg).max() if np.abs(K_neg).max() > 0 else K_neg
    
    # Combine positive and negative components
    K = K_pos - neg_weight * K_neg
    
    return K

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
    # Compute kernel matrix
    K = compute_kernel_matrix(X, kernel_type, **params)
    
    # Use t-SNE to reduce to 2D for visualization
    # t-SNE works well with pairwise similarity/distance matrices
    tsne = TSNE(n_components=2, metric='precomputed', perplexity=min(30, X.shape[0]-1))
    
    # t-SNE works with distances, so convert similarities to distances
    # We use a heuristic: D = 1 - normalized(K)
    K_normalized = K / K.max()
    D = 1 - K_normalized
    
    # Handle non-positive-semidefinite kernel
    # Add a small identity matrix to ensure it's positive definite
    if kernel_type.lower() == 'kreĭn' or kernel_type.lower() == 'krein':
        D = D + np.eye(D.shape[0]) * 0.001
    
    # Apply t-SNE
    X_transformed = tsne.fit_transform(D)
    
    return X_transformed

def apply_kernel_pca(X, kernel_type, n_components=2, **params):
    """
    Apply kernel PCA transformation to data
    
    Parameters:
    X: numpy array of shape (n_samples, n_features)
    kernel_type: str, type of kernel
    n_components: int, number of components to keep
    params: additional parameters for the kernel
    
    Returns:
    X_transformed: numpy array of shape (n_samples, n_components)
    """
    # For Kreĭn kernels, we use our own implementation
    if kernel_type.lower() == 'kreĭn' or kernel_type.lower() == 'krein':
        K = compute_kernel_matrix(X, kernel_type, **params)
        # Center the kernel matrix
        n_samples = K.shape[0]
        one_n = np.ones((n_samples, n_samples)) / n_samples
        K_centered = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
        
        # Eigendecomposition of the centered kernel matrix
        eigvals, eigvecs = np.linalg.eigh(K_centered)
        
        # Sort eigenvalues and eigenvectors in descending order of absolute eigenvalue
        idx = np.argsort(np.abs(eigvals))[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Use top n_components
        X_transformed = eigvecs[:, :n_components]
        
        return X_transformed
    
    # For standard kernels, use scikit-learn's KernelPCA
    else:
        if kernel_type.lower() == 'linear':
            kpca = KernelPCA(n_components=n_components, kernel='linear')
        elif kernel_type.lower() == 'rbf':
            gamma = params.get('gamma', 1.0)
            kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=gamma)
        elif kernel_type.lower() == 'polynomial':
            degree = params.get('degree', 3)
            gamma = params.get('gamma', 1.0)
            coef0 = params.get('coef0', 1.0)
            kpca = KernelPCA(n_components=n_components, kernel='poly', 
                             degree=degree, gamma=gamma, coef0=coef0)
        elif kernel_type.lower() == 'sigmoid':
            gamma = params.get('gamma', 1.0)
            coef0 = params.get('coef0', 1.0)
            kpca = KernelPCA(n_components=n_components, kernel='sigmoid', 
                             gamma=gamma, coef0=coef0)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
        
        X_transformed = kpca.fit_transform(X)
        return X_transformed