# We want to solve iteratively system Y = X*b using Landweber method with a twist of J-duality map. 
# We construct matrices X and b from uniform or normal distribution

import numpy as np
import pandas as pd
import kagglehub
import shutil
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import os
import sys

sys.stderr = open(os.devnull, "w")

def generate_matrix(m, n, distribution="uniform"):
    if distribution == "uniform":
        return np.random.uniform(low=0, high=1, size=(m, n))
    elif distribution == "normal":
        return np.random.normal(loc=0, scale=1, size=(m, n))
    else:
        raise ValueError("Invalid distribution type. Choose 'uniform' or 'normal'.")

def add_noise(Y, sigma, noise_type="uniform"):
    if noise_type == "uniform":
        noise = np.random.uniform(low=-sigma, high=sigma, size=Y.shape)
    elif noise_type == "normal":
        noise = np.random.normal(loc=0, scale=sigma, size=Y.shape)
    else:
        raise ValueError("Invalid noise type. Choose 'uniform' or 'normal'.")
    return Y + noise

def J_duality_map(b, p, s):
    if np.linalg.norm(b) == 0:
        return b  
    sum_p_norm = np.sum(np.power(np.abs(b), p))  
    scaling_factor = np.power(sum_p_norm, (s / p) - 1)
    return scaling_factor * np.power(np.abs(b), p - 1) * np.sign(b)

def rmse(Y, Yhat):
    return np.sqrt(mean_squared_error(Y, Yhat))

def Landweber(X, y, MAXIT, p, b_start, tol=1e-3):
    Xt = X.T
    mu = 0.1 / (np.linalg.norm(X) ** 2)
    b_new = b_start
    q = p / (p - 1) 
    s = 2
    r = s / (s - 1)
    
    rmse_list = []
    prev_rmse = float('inf')
    for _ in range(MAXIT):
        res = X @ b_new - y
        R = J_duality_map(res, p, s)
        b_new = J_duality_map(J_duality_map(b_new, p, s) - mu * Xt @ R, q, r)
        
        error = rmse(y, X @ b_new)
        rmse_list.append(error)
        
        if abs(prev_rmse - error) < tol:
            break
        prev_rmse = error
    
    return b_new, rmse_list

def optimize_p(X, Y, MAXIT):
    m, n = X.shape
    b_true = np.random.uniform(low=0, high=1, size=(n, 1))
    Y = X @ b_true
    b_start = np.random.uniform(low=0, high=1, size=(n, 1))
    
    best_p, best_b, best_rmse, best_rmse_list = None, None, float('inf'), []
    
    for p in np.arange(1.1, 2.1, 0.1):
        b_estimated, rmse_list = Landweber(X, Y, MAXIT, p, b_start)
        error = rmse(Y, X @ b_estimated)
        
        if error < best_rmse:
            best_rmse, best_p, best_b, best_rmse_list = error, p, b_estimated, rmse_list
    
    return best_p, best_b, best_rmse, best_rmse_list

def OLS_solution(X, Y):
    model = LinearRegression(fit_intercept=False)
    model.fit(X, Y)
    return model.coef_.T

def test_methods(X, Y, noise_sigma=0.1, noise_type="uniform", MAXIT=1000):
    m, n = X.shape
    print(f"\nCase: Overdetermined (m={m}, n={n})")
    
    Y_noisy = add_noise(Y, noise_sigma, noise_type)

    # Landweber method
    start_time = time.time()
    best_p, best_b, best_rmse, best_rmse_list = optimize_p(X, Y_noisy, MAXIT)
    landweber_time = time.time() - start_time

    # OLS Solution
    start_time = time.time()
    b_solution = OLS_solution(X, Y_noisy)
    solve_time = time.time() - start_time
    rmse_solution = rmse(Y, X @ b_solution)

    # Print results
    print(f"Noise type: {noise_type}, Sigma: {noise_sigma}")
    print(f"Landweber: Best p={best_p}, RMSE={best_rmse:.6f}, Time={landweber_time:.4f}s")
    print(f"OLS: RMSE={rmse_solution:.6f}, Time={solve_time:.4f}s")

    # Plot RMSE
    plt.plot(best_rmse_list)
    plt.xlabel("Iterations")
    plt.ylabel("RMSE")
    plt.title(f"RMSE Decrease Over Iterations (Overdetermined, {noise_type} noise)")
    plt.grid()
    plt.show()

# Function to load dataset from KaggleHub

def load_dataset(dataset_name):
    dataset_mapping = {
        "glass": "uciml/glass",
        "turkish_music": "joebeachcapital/turkish-music-emotion",
        "wine_quality": "yasserh/wine-quality-dataset",
        "breast_cancer": "uciml/breast-cancer-wisconsin-data",
        "magic_gamma_telescope": "abhinand05/magic-gamma-telescope-dataset"
    }
    
    if dataset_name not in dataset_mapping:
        raise ValueError("Dataset not found in predefined list.")
    
    path = kagglehub.dataset_download(dataset_mapping[dataset_name])
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in dataset folder.")
    csv_file = csv_files[0]  # Pick first CSV file
    df = pd.read_csv(os.path.join(path, csv_file))
    local_path = os.path.join(os.getcwd(), f"{dataset_name}.csv")
    shutil.copy(os.path.join(path, csv_file), local_path)
    
    X = df.iloc[:, :-1].values  # Features
    y = df.iloc[:, -1].values.reshape(-1, 1)  # Labels as column vector
    
    print(f"Dataset {dataset_name} loaded successfully and saved locally!")
    return X, y

# Example usage
X, Y = load_dataset("magic_gamma_telescope")
test_methods(X, Y)
# test_methods(generate_matrix(500, 100, distribution="uniform"))