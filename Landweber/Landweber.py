# We want to solve iteratively system Y = X*b using Landweber method with a twist of J-duality map. 
# We construct matrices X and b from uniform or normal distribution

import numpy as np
import pandas as pd
from pyparsing import col
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import os
#sys.stderr = open(os.devnull, "w")

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

def Landweber(X, y, MAXIT, p, b_start, tol=1e-6):
    Xt = X.T
    mu = 0.01 / (np.linalg.norm(X) ** 2)
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
    rows, cols = X.shape
    b_true = np.random.uniform(low=0, high=1, size=(cols, 1))
    Y = X @ b_true
    b_start = np.random.uniform(low=0, high=1, size=(cols, 1))
    
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

def test_methods(X, Y, noise_sigma=0.1, noise_type="uniform", MAXIT=5000):
    rows, cols = X.shape
    if (rows < cols):
        print(f"\nCase: Underdetermined (rows={rows}, cols={cols})")
    else:    
        print(f"\nCase: Overdetermined (rows={rows}, cols={cols})")
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
    return {"landweber_time": landweber_time,  "landweber_rmse": best_rmse, "ols_time": solve_time, "ols_rmse": rmse_solution}

def svd_solution(X, Y):    
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Compute pseudo-inverse of Sigma
    S_pinv = np.diag(1 / S)  # Invert nonzero singular values
    
    # Compute pseudo-inverse of X
    X_pinv = Vt.T @ S_pinv @ U.T
    
    # Compute solution b
    b = X_pinv @ Y
    
    return b

# Function to load dataset from KaggleHub

def load_dataset(dataset_name):
    dataset_mapping = {
        "glass_identification": 42, # mu = 0.1
        "wine_quality": 186, # mu =0.1
        "isolet": 54, # mu = 0.1
        "pen_based_recognition_of_handwritten_digits": 81, # mu = 0.001
        "airfoil_self_noise": 291, # mu = 0.01 OXI KAI TOSO KALO DATASET
        "website_phishing": 379, # mu = 0.01
         "combined_cycle_power_plant": 294 # mu = 0.01 (bgazei overflow alla kalo rmse), mu = 0.001 (den bgazei overflow alla xeirotero rmse)
    }
    
    if dataset_name not in dataset_mapping:
        raise ValueError("Dataset not found in predefined list.")
    
    dataset = fetch_ucirepo(id=dataset_mapping[dataset_name])
    X = dataset.data.features
    y = dataset.data.targets
    
    # Convert to DataFrame for saving
    df = pd.concat([X, y], axis=1)
    local_path = os.path.join(os.getcwd(), f"{dataset_name}.csv")
    df.to_csv(local_path, index=False)
    
    print(f"Dataset {dataset_name} loaded successfully and saved locally!")
    return X.values, y.values.reshape(-1, 1)

# Example usage
X, Y = load_dataset("combined_cycle_power_plant")
test_methods(X, Y)

def test_simulated_data(X,Y, add_noise_flag=True, noise_sigma = 0.1, noise_type="uniform"):
    if (add_noise_flag): 
        Y = add_noise(Y, noise_sigma, noise_type)
        
    # Landweber method
    start_time = time.time()
    best_p, best_b, best_rmse, best_rmse_list = optimize_p(X, Y, 500)
    landweber_time = time.time() - start_time

    # OLS Solution
    start_time = time.time()
    b_solution = OLS_solution(X, Y)
    solve_time = time.time() - start_time
    rmse_solution = rmse(Y, X @ b_solution)

    return {"landweber_time": landweber_time, "landweber_rmse": best_rmse, "ols_time": solve_time, "ols_rmse": rmse_solution}

def plot_benchmark_data(benchmark_data):
    """
    Plots time and RMSE comparisons for benchmark data, creating separate graphs for each column.

    Args:
        benchmark_data (dict): A nested dictionary where keys are column counts,
                               and values are dictionaries keyed by row counts,
                               containing 'landweber_time', 'ols_time', etc.
    """
    if not benchmark_data:
        print("Error: Benchmark data dictionary is empty.")
        return

    for col_val, data_for_col in benchmark_data.items():
        row_keys = sorted(data_for_col.keys())

        if not row_keys:
            print(f"Warning: No row data found for column value {col_val}. Skipping this column.")
            continue

        row_labels = row_keys  # The keys are the row counts now

        # --- Data Extraction for the current col_val ---
        landweber_times = [data_for_col[r_key]['landweber_time'] for r_key in row_keys]
        ols_times = [data_for_col[r_key]['ols_time'] for r_key in row_keys]
        landweber_rmses = [data_for_col[r_key]['landweber_rmse'] for r_key in row_keys]
        ols_rmses = [data_for_col[r_key]['ols_rmse'] for r_key in row_keys]

        # --- Plotting Setup ---
        x = np.arange(len(row_labels))  # the label locations
        bar_width = 0.35  # the width of the bars

        # --- Plot 1: Time Comparison ---
        plt.figure(figsize=(10, 6))
        plt.bar(x - bar_width/2, landweber_times, bar_width, label='Landweber Time', color='green')
        plt.bar(x + bar_width/2, ols_times, bar_width, label='OLS Time', color='blue')
        plt.xlabel('Number of Rows')
        plt.ylabel('Time (seconds)')
        plt.title(f'Time Comparison (Columns = {col_val})')
        plt.xticks(x, row_labels)
        plt.legend()
        plt.savefig(f"columns-{col_val}_time.png")
        plt.close()

        # --- Plot 2: RMSE Comparison ---
        plt.figure(figsize=(10, 6))
        plt.bar(x - bar_width/2, landweber_rmses, bar_width, label='Landweber RMSE', color='green')
        plt.bar(x + bar_width/2, ols_rmses, bar_width, label='OLS RMSE', color='blue')
        plt.xlabel('Number of Rows')
        plt.ylabel('RMSE')
        plt.title(f'RMSE Comparison (Columns = {col_val})')
        plt.xticks(x, row_labels)
        plt.legend()
        plt.savefig(f"columns-{col_val}_rmse.png")
        print("Saved RMSE and time plot for columns = ", col_val)
        plt.close()

def simulated_data():
    benchmark_data = dict()

    row_factors = [10,20,50,70,100]
    col_values = [5, 10, 20]
    distribution="uniform"
    for cols in col_values:
        benchmark_data[cols] = dict()
        for row_factor in row_factors:
            rows = row_factor*cols
            print("pair: ", (rows, cols))
            X = generate_matrix(rows, cols, distribution)
            Y = generate_matrix(rows, 1, distribution)
            benchmark_data[cols][rows] = test_simulated_data(X, Y)
    print(benchmark_data)    
    plot_benchmark_data(benchmark_data)


simulated_data()

# --- Simulated data ---
# benchmark_data = { //keys are columns
#     5: { //keys are rows -> (50,5) design matrix etc...
#         50: {'landweber_time': 9.201084852218628, 'landweber_rmse': np.float64(0.0004352763317473863), 'ols_time': 0.007337093353271484, 'ols_rmse': np.float64(0.30610172360285975)},
#         100: {'landweber_time': 6.7429139614105225, 'landweber_rmse': np.float64(3.9526075800953006e-05), 'ols_time': 0.00041604042053222656, 'ols_rmse': np.float64(0.2843897313122302)},
#         250: {'landweber_time': 6.311493158340454, 'landweber_rmse': np.float64(2.1406425335692127e-05), 'ols_time': 0.0004620552062988281, 'ols_rmse': np.float64(0.31778657266596677)},
#         350: {'landweber_time': 6.658552885055542, 'landweber_rmse': np.float64(2.6325787184349344e-05), 'ols_time': 0.0026092529296875, 'ols_rmse': np.float64(0.3122532120447918)},
#         500: {'landweber_time': 5.762638092041016, 'landweber_rmse': np.float64(1.2143653298255198e-05), 'ols_time': 0.0004801750183105469, 'ols_rmse': np.float64(0.3176851510409322)}
#     },
#     7: {
#         70: {'landweber_time': 11.598745822906494, 'landweber_rmse': np.float64(0.0002145684081750422), 'ols_time': 0.0011789798736572266, 'ols_rmse': np.float64(0.332644235722981)},
#         140: {'landweber_time': 7.850435018539429, 'landweber_rmse': np.float64(0.00013392624318127766), 'ols_time': 0.0004391670227050781, 'ols_rmse': np.float64(0.2921672192379161)},
#         350: {'landweber_time': 8.203042030334473, 'landweber_rmse': np.float64(0.0004935043620940951), 'ols_time': 0.0005908012390136719, 'ols_rmse': np.float64(0.31373158357136743)},
#         490: {'landweber_time': 8.057960033416748, 'landweber_rmse': np.float64(0.00014468370814481028), 'ols_time': 0.00047779083251953125, 'ols_rmse': np.float64(0.31251916595130963)},
#         700: {'landweber_time': 7.60652494430542, 'landweber_rmse': np.float64(3.222495894119661e-05), 'ols_time': 0.0024902820587158203, 'ols_rmse': np.float64(0.3108450054951852)}
#     },
#     10: {
#         100: {'landweber_time': 9.486278295516968, 'landweber_rmse': np.float64(0.0007396925344087137), 'ols_time': 0.0004527568817138672, 'ols_rmse': np.float64(0.29332221882929993)},
#         200: {'landweber_time': 8.699727773666382, 'landweber_rmse': np.float64(0.00017741582695153627), 'ols_time': 0.0004391670227050781, 'ols_rmse': np.float64(0.3127030477810623)},
#         500: {'landweber_time': 7.928871154785156, 'landweber_rmse': np.float64(4.388874789405075e-05), 'ols_time': 0.0007069110870361328, 'ols_rmse': np.float64(0.29682131957464886)},
#         700: {'landweber_time': 8.922483205795288, 'landweber_rmse': np.float64(9.926272179680093e-05), 'ols_time': 0.0005257129669189453, 'ols_rmse': np.float64(0.3084091669552287)},
#         1000: {'landweber_time': 9.126970052719116, 'landweber_rmse': np.float64(8.533387574285899e-05), 'ols_time': 0.0005650520324707031, 'ols_rmse': np.float64(0.29952050790767953)}
#     }
# }
