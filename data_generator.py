import numpy as np
from sklearn.datasets import make_moons, make_circles, make_classification

def generate_sample_data(data_type, n_samples=100, noise=0.1):
    """
    Generate synthetic data for visualization and exploration
    
    Parameters:
    data_type: str, type of data to generate
    n_samples: int, number of samples to generate
    noise: float, noise level in the data
    
    Returns:
    X: numpy array of shape (n_samples, n_features)
    y: numpy array of shape (n_samples,), class labels
    """
    if data_type.lower() == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    
    elif data_type.lower() == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    
    elif data_type.lower() == 'linearly_separable':
        X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, 
                                  n_informative=2, random_state=42, 
                                  n_clusters_per_class=1, class_sep=2.0)
    
    elif data_type.lower() == 'drug_response':
        X, y = generate_drug_response_data(n_samples=n_samples, noise=noise)
    
    elif data_type.lower() == 'bacteria_markers':
        X, y = generate_bacteria_markers_data(n_samples=n_samples, noise=noise)
    
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
    
    return X, y

def generate_drug_response_data(n_samples=100, noise=0.1):
    """
    Generate synthetic data representing antibiotic responses
    
    Parameters:
    n_samples: int, number of samples to generate
    noise: float, noise level in the data
    
    Returns:
    X: numpy array of shape (n_samples, 5), features representing:
       - drug A concentration
       - drug B concentration
       - target pathway inhibition
       - stress response
       - energy metabolism
    y: numpy array of shape (n_samples,), synergy class (0: antagonistic, 1: additive, 2: synergistic)
    """
    np.random.seed(42)
    
    # Generate 5 features
    X = np.zeros((n_samples, 5))
    
    # Drug A concentration
    X[:, 0] = np.random.uniform(0, 1, n_samples)
    
    # Drug B concentration
    X[:, 1] = np.random.uniform(0, 1, n_samples)
    
    # Generate synergy classes based on the drug concentrations
    # In real-world scenarios, synergy depends on complex interactions
    # Here, we simulate it with a simplified model
    
    # Base synergy score determined by interaction of drug concentrations
    synergy_score = 2.5 * X[:, 0] * X[:, 1] - 0.5 * (X[:, 0] + X[:, 1])
    
    # Determine target pathway inhibition
    # Higher synergy usually leads to higher inhibition
    X[:, 2] = 0.7 * synergy_score + 0.3 * (X[:, 0] + X[:, 1]) + np.random.normal(0, noise, n_samples)
    X[:, 2] = np.clip(X[:, 2] + 0.5, 0, 1)  # Adjust and clip to 0-1 range
    
    # Stress response - often inversely related to synergy for antibiotics
    X[:, 3] = 1 - (0.6 * synergy_score + 0.4 * np.random.uniform(0, 1, n_samples))
    X[:, 3] = np.clip(X[:, 3], 0, 1)
    
    # Energy metabolism - affected by both drugs
    X[:, 4] = 0.4 * X[:, 0] + 0.4 * X[:, 1] - 0.3 * synergy_score + np.random.normal(0, noise, n_samples)
    X[:, 4] = np.clip(X[:, 4] + 0.5, 0, 1)  # Adjust and clip to 0-1 range
    
    # Add random noise to all features
    X += np.random.normal(0, noise, X.shape)
    X = np.clip(X, 0, 1)  # Ensure all features are in 0-1 range
    
    # Assign classes based on synergy score (with some noise)
    synergy_score += np.random.normal(0, noise, n_samples)
    y = np.zeros(n_samples, dtype=int)
    y[synergy_score > 0.5] = 2  # Synergistic
    y[(synergy_score > -0.2) & (synergy_score <= 0.5)] = 1  # Additive
    y[synergy_score <= -0.2] = 0  # Antagonistic
    
    return X, y

def generate_bacteria_markers_data(n_samples=100, noise=0.1):
    """
    Generate synthetic data representing bacterial gene expression markers
    
    Parameters:
    n_samples: int, number of samples to generate
    noise: float, noise level in the data
    
    Returns:
    X: numpy array of shape (n_samples, 6), features representing different marker genes
    y: numpy array of shape (n_samples,), bacterial strain (0, 1, 2)
    """
    np.random.seed(42)
    
    # Number of samples per class
    n_per_class = n_samples // 3
    
    # Generate 6 features for each class
    X = np.zeros((n_samples, 6))
    y = np.zeros(n_samples, dtype=int)
    
    # Class 0: Drug-sensitive strain
    X[:n_per_class, 0] = np.random.normal(0.7, noise, n_per_class)  # High expression of drug target
    X[:n_per_class, 1] = np.random.normal(0.3, noise, n_per_class)  # Low efflux pump expression
    X[:n_per_class, 2] = np.random.normal(0.2, noise, n_per_class)  # Low beta-lactamase
    X[:n_per_class, 3] = np.random.normal(0.6, noise, n_per_class)  # Moderate cell wall thickness
    X[:n_per_class, 4] = np.random.normal(0.7, noise, n_per_class)  # High metabolic activity
    X[:n_per_class, 5] = np.random.normal(0.4, noise, n_per_class)  # Moderate stress response
    y[:n_per_class] = 0
    
    # Class 1: Intermediate resistance
    X[n_per_class:2*n_per_class, 0] = np.random.normal(0.4, noise, n_per_class)  # Moderate expression of drug target
    X[n_per_class:2*n_per_class, 1] = np.random.normal(0.6, noise, n_per_class)  # Moderate efflux pump expression
    X[n_per_class:2*n_per_class, 2] = np.random.normal(0.5, noise, n_per_class)  # Moderate beta-lactamase
    X[n_per_class:2*n_per_class, 3] = np.random.normal(0.7, noise, n_per_class)  # Thicker cell wall
    X[n_per_class:2*n_per_class, 4] = np.random.normal(0.5, noise, n_per_class)  # Moderate metabolic activity
    X[n_per_class:2*n_per_class, 5] = np.random.normal(0.6, noise, n_per_class)  # Higher stress response
    y[n_per_class:2*n_per_class] = 1
    
    # Class 2: Highly resistant strain
    X[2*n_per_class:, 0] = np.random.normal(0.2, noise, n_samples-2*n_per_class)  # Low expression of drug target
    X[2*n_per_class:, 1] = np.random.normal(0.8, noise, n_samples-2*n_per_class)  # High efflux pump expression
    X[2*n_per_class:, 2] = np.random.normal(0.9, noise, n_samples-2*n_per_class)  # High beta-lactamase
    X[2*n_per_class:, 3] = np.random.normal(0.8, noise, n_samples-2*n_per_class)  # Thick cell wall
    X[2*n_per_class:, 4] = np.random.normal(0.3, noise, n_samples-2*n_per_class)  # Low metabolic activity
    X[2*n_per_class:, 5] = np.random.normal(0.8, noise, n_samples-2*n_per_class)  # High stress response
    y[2*n_per_class:] = 2
    
    # Add random noise and clip to 0-1 range
    X = np.clip(X, 0, 1)
    
    # Shuffle data
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    
    return X, y