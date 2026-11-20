"""纯标准库线性代数与高斯分布工具。

只依赖 math，便于在无 numpy/sklearn 的受限环境运行；
矩阵以“行的列表”表示，向量为 float 列表。
"""
import math


def mat_zeros(n, m):
    return [[0.0] * m for _ in range(n)]


def mat_copy(a):
    return [row[:] for row in a]


def mat_identity(n):
    a = mat_zeros(n, n)
    for i in range(n):
        a[i][i] = 1.0
    return a


def mat_transpose(a):
    return [list(col) for col in zip(*a)]


def mat_vec(a, v):
    return [sum(a[i][j] * v[j] for j in range(len(v))) for i in range(len(a))]


def cholesky(a):
    """正定矩阵的 Cholesky 分解，返回下三角 L 使 L L^T = A。"""
    n = len(a)
    L = mat_zeros(n, n)
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                d = a[i][i] - s
                if d <= 1e-12:
                    raise ValueError('协方差矩阵非正定，无法完成分解')
                L[i][j] = math.sqrt(d)
            else:
                L[i][j] = (a[i][j] - s) / L[j][j]
    return L


def solve_triangular(L, b, lower=True):
    """解三角方程组 L x = b（lower=True）或 L^T x = b。"""
    n = len(L)
    x = [0.0] * n
    if lower:
        for i in range(n):
            s = sum(L[i][j] * x[j] for j in range(i))
            x[i] = (b[i] - s) / L[i][i]
    else:
        for i in range(n - 1, -1, -1):
            s = sum(L[j][i] * x[j] for j in range(i + 1, n))
            x[i] = (b[i] - s) / L[i][i]
    return x


def solve_spd(a, b):
    """解正定方程组 A x = b，b 可为向量或矩阵（按列求解）。"""
    L = cholesky(a)
    if b and isinstance(b[0], list):
        cols = mat_transpose(b)
        out = [solve_triangular(L, solve_triangular(L, c), lower=False) for c in cols]
        return mat_transpose(out)
    y = solve_triangular(L, b)
    return solve_triangular(L, y, lower=False)


def logdet_spd(a):
    L = cholesky(a)
    return 2.0 * sum(math.log(L[i][i]) for i in range(len(L)))


def sub_matrix(a, rows, cols):
    return [[a[i][j] for j in cols] for i in rows]


def sub_vector(v, idx):
    return [v[i] for i in idx]


def gaussian_logpdf(x, mean, cov):
    """多元高斯对数密度，维度由传入的协方差决定。"""
    d = len(mean)
    diff = [x[i] - mean[i] for i in range(d)]
    L = cholesky(cov)
    sol = solve_triangular(L, solve_triangular(L, diff), lower=False)
    quad = sum(diff[i] * sol[i] for i in range(d))
    ld = 2.0 * sum(math.log(L[i][i]) for i in range(d))
    return -0.5 * (d * math.log(2.0 * math.pi) + ld + quad)


def conditional_normal(mean, cov, observed_idx, observed_vals, missing_idx):
    """给定实测维度后的缺项条件分布 N(mu_m|o, Sigma_m|o)。

    返回 (条件均值向量, 条件协方差矩阵)，均按 missing_idx 顺序。
    """
    if not missing_idx:
        return [], []
    if not observed_idx:
        return sub_vector(mean, missing_idx), sub_matrix(cov, missing_idx, missing_idx)
    cov_mm = sub_matrix(cov, missing_idx, missing_idx)
    cov_mo = sub_matrix(cov, missing_idx, observed_idx)
    cov_oo = sub_matrix(cov, observed_idx, observed_idx)
    mu_o = sub_vector(mean, observed_idx)
    mu_m = sub_vector(mean, missing_idx)
    diff = [observed_vals[k] - mu_o[k] for k in range(len(observed_idx))]
    # beta = Sigma_mo Sigma_oo^{-1}
    beta = mat_transpose(solve_spd(cov_oo, mat_transpose(cov_mo)))
    cond_mean = [mu_m[i] + sum(beta[i][k] * diff[k] for k in range(len(diff)))
                 for i in range(len(missing_idx))]
    cond_cov = [[cov_mm[i][j] - sum(beta[i][k] * cov_mo[j][k] for k in range(len(observed_idx)))
                 for j in range(len(missing_idx))] for i in range(len(missing_idx))]
    # 消除浮点噪声，强制对称
    for i in range(len(cond_cov)):
        for j in range(i):
            v = 0.5 * (cond_cov[i][j] + cond_cov[j][i])
            cond_cov[i][j] = cond_cov[j][i] = v
    return cond_mean, cond_cov
