import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons, make_classification

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
    if data_type == "Two Clusters":
        X, y = make_blobs(
            n_samples=n_samples, 
            centers=2, 
            n_features=2, 
            random_state=42,
            cluster_std=noise*5
        )
    
    elif data_type == "Circles":
        X, y = make_circles(
            n_samples=n_samples, 
            noise=noise, 
            factor=0.5, 
            random_state=42
        )
    
    elif data_type == "Moons":
        X, y = make_moons(
            n_samples=n_samples, 
            noise=noise, 
            random_state=42
        )
    
    elif data_type == "Linearly Separable":
        X, y = make_classification(
            n_samples=n_samples, 
            n_features=2, 
            n_redundant=0, 
            n_informative=2,
            random_state=42, 
            n_clusters_per_class=1,
            class_sep=1.5 - noise
        )
    
    else:
        # Default to blobs
        X, y = make_blobs(
            n_samples=n_samples, 
            centers=2, 
            n_features=2, 
            random_state=42
        )
    
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
    
    # Generate drug concentrations
    drug_A = np.random.uniform(0, 1, n_samples)
    drug_B = np.random.uniform(0, 1, n_samples)
    
    # Generate target pathway inhibition (nonlinear function of drug concentrations)
    pathway_inhibition = 0.5 * drug_A + 0.7 * drug_B + 1.5 * drug_A * drug_B + noise * np.random.randn(n_samples)
    
    # Generate stress response (complex function of drug concentrations)
    stress_response = 0.3 * drug_A**2 + 0.4 * drug_B**2 - 0.2 * np.sin(drug_A * 3) * np.cos(drug_B * 2) + noise * np.random.randn(n_samples)
    
    # Generate energy metabolism effect (another nonlinear function)
    energy_metabolism = 0.6 * np.exp(-2 * drug_A) + 0.4 * np.exp(-1.5 * drug_B) + 0.8 * drug_A * drug_B + noise * np.random.randn(n_samples)
    
    # Stack features
    X = np.column_stack([drug_A, drug_B, pathway_inhibition, stress_response, energy_metabolism])
    
    # Generate synergy classes
    synergy_score = 0.3 * pathway_inhibition + 0.4 * stress_response - 0.3 * energy_metabolism
    y = np.zeros(n_samples)
    y[synergy_score > 0.5] = 2  # Synergistic
    y[(synergy_score > 0) & (synergy_score <= 0.5)] = 1  # Additive
    y[synergy_score <= 0] = 0  # Antagonistic
    
    return X, y
