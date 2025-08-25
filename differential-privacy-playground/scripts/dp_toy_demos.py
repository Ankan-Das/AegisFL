import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

# ------------------ 1) Sensitivity (count and mean) ------------------

def sensitivity_count():
    D1 = [1, 0, 1, 1, 0]    # 3 likes
    D2 = [1, 0, 1, 1, 1]    # 4 likes
    f = lambda D: sum(D)
    diff = abs(f(D1) - f(D2))
    print(f"[Sensitivity: count] f(D1) = {f(D1)}, f(D2) = {f(D2)}, Δ = {diff} (should be 1)")
    
def sensitivity_mean_01(n=100):
    # Max swing if values in [0,1]: Δ = 1/n
    print(f"[Sensitivity: mean] Δ = 1/n = {1/n}")


# ------------------ 2) Laplace Mechanism ------------------
def laplace_mech(true_value, eps, sens):
    scale = sens / eps
    return true_value + np.random.laplace(0, scale)

def demo_laplace(true_count=42, eps=1.0, sens=1.0, trials=2000):
    outs = [laplace_mech(true_value=true_count, eps=eps, sens=sens) for _ in range(trials)]
    print(f"[Laplace] ε={eps}, scale={sens/eps:.3f}, mean={np.mean(outs):.2f}, std={np.std(outs):.2f}")
    plt.figure()
    plt.hist(outs, bins=40)
    plt.title(f"Laplace noisy count (ε={eps})")
    plt.xlabel("noisy count"); plt.ylabel("freq")
    plt.savefig(OUT / f"laplace_hist_e{eps}.png", dpi=120)
    plt.close()
    

# ------------------ 3) Gaussian Mechanism ------------------
def gaussian_mech(true_value, eps, delta, sens):
    sigma = np.sqrt(2*np.log(1.25/delta)) * (sens / eps)
    return true_value + np.random.normal(0, sigma)

def demo_gaussian(true_count=42, eps=1.0, delta=1e-5, sens=1.0, trials=2000):
    sigma = np.sqrt(2*np.log(1.25/delta)) * (sens / eps)
    outs = [true_count + np.random.normal(0, sigma) for _ in range(trials)]
    print(f"[Gaussian] ε={eps}, δ={delta}, σ={sigma:.3f}, mean={np.mean(outs):.2f}, std={np.std(outs):.2f}")
    plt.figure()
    plt.hist(outs, bins=40)
    plt.title(f"Gaussian noisy count (ε={eps}, δ={delta})")
    plt.xlabel("noisy count"); plt.ylabel("freq")
    plt.savefig(OUT / f"gaussian_hist_e{eps}_d{delta}.png", dpi=120)
    plt.close()
    

# ------------------ 4) Composition Demo ------------------
def composition_demo(eps_per_query=0.5, num_queries=4, delta_per_query=1e-5):
    eps_total = eps_per_query * num_queries
    delta_total = delta_per_query * num_queries
    print(f"[Composition] {num_queries} queries at ε={eps_per_query} → ε_total={eps_total}")
    print(f"[Composition] δ_total = {delta_total}")
    

# ------------------ 5) Bonus: indistinguisibility check ------------------
def indistinguishability_demo(n=1000, eps=1.0, trials=5000):
    # Two neighbouring datasets differ by one person's value in [0,1]
    mu1 = 0.5
    mu2 = mu1 + 1.0/n
    sens = 1.0/n
    scale = sens / eps  # Laplace
    s1 = mu1 + np.random.laplace(0, scale, size=trials)
    s2 = mu2 + np.random.laplace(0, scale, size=trials)
    t = 0.5  # threshold
    p1 = (s1 <= t).mean() + 1e-12
    p2 = (s2 <= t).mean() + 1e-12
    ratio = p1 / p2
    print(f"[Indistinguishability] P(M(D)≤{t})={p1:.3f}, P(M(D')≤{t})={p2:.3f}, ratio≈{ratio:.3f}, e^ε={np.exp(eps):.3f}")
    

if __name__ == "__main__":
    sensitivity_count()
    sensitivity_mean_01(n=1000)
    print("-"*50)
    demo_laplace(true_count=42, eps=1.0, sens=1.0)
    demo_laplace(true_count=42, eps=0.5, sens=1.0)
    print("-"*50)
    demo_gaussian(true_count=42, eps=1.0, delta=1e-5, sens=1.0)
    print("-"*50)
    composition_demo(eps_per_query=0.5, num_queries=4, delta_per_query=1e-6)
    print("-"*50)
    indistinguishability_demo(n=1000, eps=1.0, trials=5000)
