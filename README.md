# How to run this code

```bash
python3.10 -m venv env #or whichever python you have installed
source env/bin/activate
pip install -r Landweber/requirements.txt

python3 Landweber/landweber.py
```

## Algorithm's complexity

Let us that the design matrix $X \in \mathbb{R}^{n \times m}$. Then $\beta \in \mathbb{R}^{m\times 1}$ and $y\in\mathbb{R}^{n\times 1}$. Firstly, the computational complexity of the duality map is linear with respect to its input, therefore $O(m)$. The computational complexity per Landweber iteration is primarily determined by matrix-vector multiplications and duality mappings. The computation of the residual, $Xb - y$ is $O(mn)$, the duality mapping applied to it is of linear complexity with respect to its input, therefore line 6 has complexity $O(mn)$. The same holds true for line 7. The first duality map is applied with input of dimensions $m\times 1$ and is of complexity $O(m)$, the $\mu \cdot X^T \cdot R$ is of complexity $O(mn)$, therefore $J\_duality\_map(J\_duality\_map(\beta_{\text{old}}, p, s) - \mu \cdot X^T \cdot R, q, r)$ is of complexity $O(mn)$.

For a maximum of MAXIT iterations, the worst-case complexity of the algorithm is $O(\text{MAXIT} \cdot n \cdot m)$. Early stopping based on the convergence of the root mean square error (RMSE) can reduce the effective number of iterations $k$, leading to an improved practical complexity of $O(k \cdot n \cdot m)$.

## Real World Datasets

    The algorithm was tested on several real-world datasets. The Landweber iterative method was applied to solve the linear system $y = X\beta + \varepsilon$. All datasets exhibited highly overdetermined design matrices, reflecting real-world scenarios where observations vastly outnumber predictors. Uniform noise ($\sigma=0.1$) was introduced the target variable $y$ to simulate real-world imperfections. The method ran for a maximum of $5000$ iterations, ensuring convergence within a tolerance of $10^{-6}$.

- Glass Identification Dataset - URL: https://www.kaggle.com/datasets/uciml/glass
- Wine Quality Dataset - URL: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
- Isolet Dataset - URL: https://www.kaggle.com/datasets/gorangsolanki/isolet-dataset
- Pen-Based Recognition of Handwritten Digits - URL: https://www.kaggle.com/datasets/duygujones/pen-based-handwritten-digit
- Airfoil Self Noise - URL: https://www.kaggle.com/datasets/fedesoriano/airfoil-selfnoise-dataset
- Website Phishing - URL: https://www.kaggle.com/datasets/akashkr/phishing-website-dataset
- Combined Cycle Power Plant - URL: https://www.kaggle.com/datasets/gova26/airpressure

## Simulated Data Benchmarks

For further evaluation of the Landweber method, simulated data was generated with varying matrix dimensions, testing the performance across different configurations. The design matrices $X$ and response vectors $y$ were constructed using a uniform distribution, with column sizes fixed at $5, 7, 10$, and $20$, and row sizes scaled as multiples $(10, 20, 50, 70, 100)$ of the columns, resulting in overdetermined systems (e.g., $50 rows \times 5$ columns, up to $2000 \times 20$ columns). Uniform noise ($\sigma = 0.1$) was added to the response vector. The Landweber method was run for a maximum of $500$ iterations, optimizing the parameter $p$ in the range $[1.1,2.1]$ to minimize RMSE, while OLS provided a baseline for comparison.

## Conclusions

The results reveal that Landweber consistently outperforms OLS in terms of RMSE across all simulated configurations in terms of RMSE. However, Landweber’s execution time is significantly higher, often exceeding $10$ seconds, compared to OLS, which completes in under $0.01$ seconds. This trend holds across both real and simulated data, highlighting Landweber’s key advantage: its ability to achieve superior accuracy, making it ideal for applications where precision is critical, such as predictive maintenance in power plants or phishing detection. However, the method’s computational cost poses a drawback, particularly for large-scale or time-sensitive applications. Overall, the Landweber method offers a compelling trade-off for scenarios prioritizing accuracy over speed, but its scalability remains a challenge for real-time systems.
