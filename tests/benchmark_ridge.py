"""Benchmark sklearn vs cuML Ridge regression."""

import time
import numpy as np
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.model_selection import cross_val_score
from cuml.linear_model import Ridge as CumlRidge
import cupy as cp


def benchmark_fit(X, y, alpha, n_runs=5):
    """Benchmark fitting time for both implementations."""
    # sklearn
    sklearn_times = []
    for _ in range(n_runs):
        model = SklearnRidge(alpha=alpha)
        start = time.perf_counter()
        model.fit(X, y)
        sklearn_times.append(time.perf_counter() - start)
    sklearn_coef = model.coef_.copy()
    sklearn_pred = model.predict(X)

    # cuML - convert to cupy arrays for GPU
    X_gpu = cp.asarray(X)
    y_gpu = cp.asarray(y)

    cuml_times = []
    for _ in range(n_runs):
        model = CumlRidge(alpha=alpha)
        start = time.perf_counter()
        model.fit(X_gpu, y_gpu)
        cp.cuda.Stream.null.synchronize()  # ensure GPU ops complete
        cuml_times.append(time.perf_counter() - start)
    cuml_coef = cp.asnumpy(model.coef_)
    cuml_pred = cp.asnumpy(model.predict(X_gpu))

    return {
        "sklearn_time_mean": np.mean(sklearn_times),
        "sklearn_time_std": np.std(sklearn_times),
        "cuml_time_mean": np.mean(cuml_times),
        "cuml_time_std": np.std(cuml_times),
        "speedup": np.mean(sklearn_times) / np.mean(cuml_times),
        "coef_max_diff": np.max(np.abs(sklearn_coef - cuml_coef)),
        "coef_correlation": np.corrcoef(sklearn_coef.flatten(), cuml_coef.flatten())[0, 1],
        "pred_max_diff": np.max(np.abs(sklearn_pred - cuml_pred)),
        "pred_correlation": np.corrcoef(sklearn_pred, cuml_pred)[0, 1],
    }


def benchmark_cv(X, y, alpha, cv_folds=5):
    """Benchmark cross-validation."""
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=cv_folds, shuffle=False)

    # sklearn CV
    start = time.perf_counter()
    sklearn_scores = cross_val_score(
        SklearnRidge(alpha=alpha), X, y, cv=cv_folds, scoring="r2"
    )
    sklearn_time = time.perf_counter() - start

    # cuML CV - manual implementation
    X_gpu = cp.asarray(X)
    y_gpu = cp.asarray(y)

    start = time.perf_counter()
    cuml_scores = []
    for train_idx, test_idx in kf.split(X):
        X_train = X_gpu[train_idx]
        X_test = X_gpu[test_idx]
        y_train = y_gpu[train_idx]
        y_test = y_gpu[test_idx]

        model = CumlRidge(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # R² score
        ss_res = cp.sum((y_test - y_pred) ** 2)
        ss_tot = cp.sum((y_test - cp.mean(y_test)) ** 2)
        r2 = 1 - ss_res / ss_tot
        cuml_scores.append(float(cp.asnumpy(r2)))

    cp.cuda.Stream.null.synchronize()
    cuml_time = time.perf_counter() - start
    cuml_scores = np.array(cuml_scores)

    return {
        "sklearn_cv_time": sklearn_time,
        "cuml_cv_time": cuml_time,
        "cv_speedup": sklearn_time / cuml_time,
        "sklearn_r2_mean": sklearn_scores.mean(),
        "cuml_r2_mean": cuml_scores.mean(),
        "r2_diff": abs(sklearn_scores.mean() - cuml_scores.mean()),
    }


def run_benchmarks():
    np.random.seed(42)

    # Test different data sizes (typical for LLM activation probing)
    configs = [
        {"n_samples": 1000, "n_features": 4096},
        {"n_samples": 5000, "n_features": 4096},
        {"n_samples": 10000, "n_features": 4096},
        {"n_samples": 20000, "n_features": 4096},
        {"n_samples": 10000, "n_features": 8192},
    ]

    alpha = 1.0

    print("=" * 80)
    print("Ridge Regression Benchmark: sklearn vs cuML")
    print("=" * 80)

    for config in configs:
        n_samples = config["n_samples"]
        n_features = config["n_features"]

        print(f"\n{'='*60}")
        print(f"Data shape: ({n_samples}, {n_features})")
        print("=" * 60)

        # Generate synthetic data
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        true_coef = np.random.randn(n_features).astype(np.float32)
        y = X @ true_coef + np.random.randn(n_samples).astype(np.float32) * 0.1

        # Benchmark fit
        print("\n[Fit benchmark]")
        fit_results = benchmark_fit(X, y, alpha)
        print(f"  sklearn: {fit_results['sklearn_time_mean']*1000:.1f}ms ± {fit_results['sklearn_time_std']*1000:.1f}ms")
        print(f"  cuML:    {fit_results['cuml_time_mean']*1000:.1f}ms ± {fit_results['cuml_time_std']*1000:.1f}ms")
        print(f"  Speedup: {fit_results['speedup']:.2f}x")
        print(f"\n[Result comparison]")
        print(f"  Coef max diff:    {fit_results['coef_max_diff']:.2e}")
        print(f"  Coef correlation: {fit_results['coef_correlation']:.10f}")
        print(f"  Pred max diff:    {fit_results['pred_max_diff']:.2e}")
        print(f"  Pred correlation: {fit_results['pred_correlation']:.10f}")

        # Benchmark CV (only for smaller sizes to keep runtime reasonable)
        if n_samples <= 10000:
            print("\n[Cross-validation benchmark (5-fold)]")
            cv_results = benchmark_cv(X, y, alpha)
            print(f"  sklearn: {cv_results['sklearn_cv_time']:.2f}s")
            print(f"  cuML:    {cv_results['cuml_cv_time']:.2f}s")
            print(f"  Speedup: {cv_results['cv_speedup']:.2f}x")
            print(f"  sklearn R² mean: {cv_results['sklearn_r2_mean']:.6f}")
            print(f"  cuML R² mean:    {cv_results['cuml_r2_mean']:.6f}")
            print(f"  R² difference:   {cv_results['r2_diff']:.2e}")


if __name__ == "__main__":
    run_benchmarks()
