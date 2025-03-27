# We want to solve iteratively system Y = X*b using landweber method with a twist of J-duality map. 
# We construct matrices X and b from uniform distribution

import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def uniform(m, n):
    return np.random.uniform(low=0, high=1, size=(m, n))

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

def Landweber(X, y, MAXIT, p, b_start, tol=1e-6):
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

# Main testing function
def test_methods(noise_sigma=0.1, noise_type="uniform", MAXIT=500):
    m = 1000
    n = 50
    X = uniform(m, n)
    b_true = np.random.uniform(low=0, high=1, size=(n, 1))
    Y = X @ b_true
    Y_noisy = add_noise(Y, noise_sigma, noise_type)
    
    # Landweber method
    start_time = time.time()
    best_p, best_b, best_rmse, best_rmse_list = optimize_p(X, Y_noisy, MAXIT)
    landweber_time = time.time() - start_time
    
    # OLS method
    start_time = time.time()
    b_ols = OLS_solution(X, Y_noisy)
    ols_time = time.time() - start_time
    
    rmse_ols = rmse(Y, X @ b_ols)
    
    # Print results
    print(f"Noise type: {noise_type}, Sigma: {noise_sigma}")
    print(f"Landweber: Best p={best_p}, RMSE={best_rmse:.6f}, Time={landweber_time:.4f}s")
    print(f"OLS: RMSE={rmse_ols:.6f}, Time={ols_time:.4f}s")
    
    # Plot RMSE
    plt.plot(best_rmse_list)
    plt.xlabel("Iterations")
    plt.ylabel("RMSE")
    plt.title(f"RMSE Decrease Over Iterations ({noise_type} noise)")
    plt.grid()
    plt.show()

# Run tests
for noise_type in ["uniform", "normal"]:
    test_methods(noise_sigma=0.1, noise_type=noise_type)