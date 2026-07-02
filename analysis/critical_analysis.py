import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-12

# -----------------------------
# 1) Omega(t): normalized angular momentum (2D)
# -----------------------------
def velocities_from_positions(X, dt=1.0):
    # X: (T,N,2)
    return (X[1:] - X[:-1]) / dt  # (T-1,N,2)

def normalized_spin_omega_2d(X, dt=1.0, remove_translation=True):
    """
    Omega(t) = sum_i (r_i x u_i) / sum_i (|r_i||u_i|)
    X: (T,N,2)
    Returns Omega: (T-1,)
    """
    V = velocities_from_positions(X, dt=dt)   # (T-1,N,2)
    Xmid = X[:-1]                              # align with V

    xcm = Xmid.mean(axis=1, keepdims=True)     # (T-1,1,2)
    R = Xmid - xcm                              # (T-1,N,2)

    if remove_translation:
        Vcm = V.mean(axis=1, keepdims=True)
        U = V - Vcm
    else:
        U = V

    cross = R[..., 0]*U[..., 1] - R[..., 1]*U[..., 0]  # (T-1,N)
    num = cross.sum(axis=1)
    den = (np.linalg.norm(R, axis=-1) * np.linalg.norm(U, axis=-1)).sum(axis=1) + EPS
    return num / den


# -----------------------------
# 2) Windowed chi = N*Var(x)
# -----------------------------
def windowed_chi(x, N, W, step):
    T = len(x)
    centers, mu, var, chi = [], [], [], []
    for start in range(0, T - W + 1, step):
        seg = x[start:start+W]
        centers.append(start + W//2)
        m = seg.mean()
        v = seg.var(ddof=0)
        mu.append(m)
        var.append(v)
        chi.append(N * v)
    return np.asarray(centers), np.asarray(mu), np.asarray(var), np.asarray(chi)


# -----------------------------
# 3) Autocorr (FFT) + tau_int
# -----------------------------
def autocorr_fft(x):
    """
    Normalized autocorrelation ac[tau] with ac[0]=1.
    """
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    if n < 2:
        return np.array([1.0])

    nfft = 1 << (2*n - 1).bit_length()
    fx = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(fx * np.conj(fx), n=nfft)[:n]
    return ac / (ac[0] + EPS)

def tau_int_from_ac(ac, cutoff="first_nonpositive"):
    """
    tau_int = 1 + 2 * sum_{tau=1..Tcut} ac[tau]
    cutoff:
      - "first_nonpositive": ac[tau] <= 0 で打ち切り（推奨）
      - "full": 全部足す（ノイズで暴れがち）
    """
    if len(ac) < 2:
        return 0.0

    if cutoff == "first_nonpositive":
        tcut = len(ac) - 1
        for t in range(1, len(ac)):
            if ac[t] <= 0:
                tcut = t - 1
                break
        s = ac[1:tcut+1].sum() if tcut >= 1 else 0.0
        return float(1.0 + 2.0 * s)

    if cutoff == "full":
        return float(1.0 + 2.0 * ac[1:].sum())

    raise ValueError("unknown cutoff")

def windowed_tau_int(x, W, step, cutoff="first_nonpositive"):
    T = len(x)
    centers, taus = [], []
    for start in range(0, T - W + 1, step):
        seg = x[start:start+W]
        ac = autocorr_fft(seg)
        tau = tau_int_from_ac(ac, cutoff=cutoff)
        centers.append(start + W//2)
        taus.append(tau)
    return np.asarray(centers), np.asarray(taus)


# -----------------------------
# 4) Utility: Pearson corr + power-law fit
# -----------------------------
def pearson_corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a0 = a[m] - a[m].mean()
    b0 = b[m] - b[m].mean()
    return float((a0*b0).sum() / (np.sqrt((a0*a0).sum())*np.sqrt((b0*b0).sum()) + EPS))

def fit_powerlaw_tau_vs_chi(chi, tau, label="Omega"):
    chi = np.asarray(chi, float)
    tau = np.asarray(tau, float)
    m = np.isfinite(chi) & np.isfinite(tau) & (chi > 0) & (tau > 0)
    if m.sum() < 3:
        return np.nan, np.nan, np.nan, int(m.sum())

    x = np.log10(chi[m])
    y = np.log10(tau[m])

    alpha, a = np.polyfit(x, y, 1)  # y = a + alpha x
    yhat = a + alpha * x
    r2 = 1.0 - ((y - yhat)**2).sum() / ((y - y.mean())**2).sum()
    pref = 10**a

    # plot
    plt.figure()
    plt.scatter(chi[m], tau[m], s=30)
    xs = np.logspace(x.min(), x.max(), 200)
    plt.plot(xs, pref * (xs**alpha))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(f"chi_{label}")
    plt.ylabel(f"tau_int_{label}")
    plt.title(f"tau_int ~ chi^alpha  (alpha={alpha:.3f}, R^2={r2:.3f}, n={m.sum()})")
    plt.show()

    return float(alpha), float(pref), float(r2), int(m.sum())

def windowed_flip_rate(x, W, step, eps=1e-8):
    """
    flip rate within each window:
      count sign changes of x(t) after removing near-zero values.
    Returns centers, flip_rate (per step)
    """
    x = np.asarray(x, float)
    T = len(x)

    centers, rates = [], []
    for start in range(0, T - W + 1, step):
        seg = x[start:start+W]

        # avoid noisy sign around 0: ignore |x|<eps by forward-fill sign
        s = np.sign(seg)
        s[np.abs(seg) < eps] = 0

        # forward fill zeros
        for i in range(1, len(s)):
            if s[i] == 0:
                s[i] = s[i-1]
        # if leading zeros remain, set to first nonzero (or +1)
        if s[0] == 0:
            nz = np.flatnonzero(s)
            s[0] = s[nz[0]] if len(nz) else 1
            for i in range(1, len(s)):
                if s[i] == 0:
                    s[i] = s[i-1]

        nflip = np.sum(s[1:] * s[:-1] < 0)
        centers.append(start + W//2)
        rates.append(nflip / (W-1))
    return np.asarray(centers), np.asarray(rates)

# -----------------------------
# 5) MAIN
# -----------------------------
# Load (あなたのデータ形式に合わせて最小限だけ残す)
data = np.load("rec_stats_300.npz")   # pos: (T,2,N)
X = data["pos"]
# X = np.load("trajectory12.npy")
X = np.transpose(X, (0, 2, 1))              # -> (T,N,2)

dt = 1.0
W = 2000
step = 660

Omega = normalized_spin_omega_2d(X, dt=dt, remove_translation=True)

N = X.shape[1]
cent_chi, mu_O, var_O, chi_O = windowed_chi(Omega, N=N, W=W, step=step)
cent_tau, tau_O = windowed_tau_int(Omega, W=W, step=step, cutoff="first_nonpositive")

# 窓中心一致チェック（基本一致するはず）
if not np.all(cent_chi == cent_tau):
    L = min(len(chi_O), len(tau_O))
    chi_O = chi_O[:L]
    tau_O = tau_O[:L]
    cent_chi = cent_chi[:L]
    cent_tau = cent_tau[:L]

print("Omega: mean=", float(Omega.mean()), "var=", float(Omega.var()))
print("chi_Omega: mean=", float(np.nanmean(chi_O)), "var=", float(np.nanvar(chi_O)))
print("tau_int_Omega: mean=", float(np.nanmean(tau_O)), "var=", float(np.nanvar(tau_O)))
print("corr(chi_Omega, tau_int_Omega) =", pearson_corr(chi_O, tau_O))

# scatter (linear)
plt.figure()
plt.scatter(chi_O, tau_O, s=30)
plt.xlabel("chi_Omega")
plt.ylabel("tau_int_Omega")
plt.title("Critical slowing down (windowed): chi vs tau_int")
plt.show()

# power-law fit (log-log)
alpha, pref, r2, npts = fit_powerlaw_tau_vs_chi(chi_O, tau_O, label="Omega")
print("alpha =", alpha, "prefactor =", pref, "R^2 =", r2, "npts =", npts)

m = np.isfinite(chi_O) & np.isfinite(tau_O) & (chi_O > 0) & (tau_O > 0)
lx = np.log10(chi_O[m])
ly = np.log10(tau_O[m])

print("log10(chi) range =", float(lx.max() - lx.min()))
print("log10(tau) range =", float(ly.max() - ly.min()))
print("n =", int(m.sum()))

print("tau_max =", float(np.nanmax(tau_O)))
print("tau_max / W =", float(np.nanmax(tau_O) / W))

# --- flip rate ---
cent_fr, flip_rate = windowed_flip_rate(Omega, W=W, step=step, eps=1e-6)

# 念のため中心合わせ（通常一致）
if not np.all(cent_fr == cent_chi):
    L = min(len(flip_rate), len(chi_O), len(tau_O))
    flip_rate = flip_rate[:L]
    chi_O = chi_O[:L]
    tau_O = tau_O[:L]

print("flip_rate: mean=", float(np.nanmean(flip_rate)), "max=", float(np.nanmax(flip_rate)))

# 1) flip_rate と (chi,tau) の相関
print("corr(flip_rate, chi_Omega) =", pearson_corr(flip_rate, chi_O))
print("corr(flip_rate, tau_int_Omega) =", pearson_corr(flip_rate, tau_O))

# 2) 上位/下位で分割してフィット（ここが本命）
m = np.isfinite(flip_rate) & np.isfinite(chi_O) & np.isfinite(tau_O) & (chi_O > 0) & (tau_O > 0)
fr = flip_rate[m]; chi = chi_O[m]; tau = tau_O[m]

# 例えば上位30%を「スイッチング窓」とする（固定値でもOK）
q = 0.80
thr = np.quantile(fr, q)
hi = fr >= thr
lo = fr <  thr

print(f"threshold flip_rate (q={q}) =", float(thr), "  n_hi =", int(hi.sum()), " n_lo =", int(lo.sum()))

# それぞれ log-log fit
alpha_hi, pref_hi, r2_hi, n_hi = fit_powerlaw_tau_vs_chi(chi[hi], tau[hi], label="Omega (high flip)")
alpha_lo, pref_lo, r2_lo, n_lo = fit_powerlaw_tau_vs_chi(chi[lo], tau[lo], label="Omega (low flip)")

print("HIGH flip: alpha =", alpha_hi, "R^2 =", r2_hi, "n =", n_hi)
print("LOW  flip: alpha =", alpha_lo, "R^2 =", r2_lo, "n =", n_lo)

# 3) 直感用：色分け散布図（線形）
plt.figure()
plt.scatter(chi[lo], tau[lo], s=30, label="low flip")
plt.scatter(chi[hi], tau[hi], s=30, label="high flip")
plt.xlabel("chi_Omega")
plt.ylabel("tau_int_Omega")
plt.title("chi vs tau_int colored by flip_rate")
plt.legend()
plt.show()



# ---- optional: |Omega| を見たければここだけON ----
# S = np.abs(Omega)
# _, _, _, chi_S = windowed_chi(S, N=N, W=W, step=step)
# _, tau_S = windowed_tau_int(S, W=W, step=step, cutoff="first_nonpositive")
# print("corr(chi_|Omega|, tau_int_|Omega|) =", pearson_corr(chi_S, tau_S))
# fit_powerlaw_tau_vs_chi(chi_S, tau_S, label="|Omega|")