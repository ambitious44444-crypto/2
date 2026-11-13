import io
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import streamlit as st

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive Multi-Gaussian Gel Fitter",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Adaptive Multi-Gaussian Gel Profile Fitter")
st.markdown(
    "Upload ImageJ lane profile CSV files (2 columns: `pixel`, `intensity`) to perform "
    "**linear-baseline + multi-Gaussian nonlinear least squares fitting**, with automatic "
    "**AICc model selection (2-peak vs. 3-peak bridging detection)** and parameter extraction ($A$, $\mu$, $\sigma$, FWHM)."
)

# ── Mathematical Models & AICc ───────────────────────────────────────────────
def multi_gaussian_linear_bg(x, a, b, *params):
    """Linear baseline (a*x + b) plus sum of N Gaussian peaks."""
    y = a * x + b
    for i in range(0, len(params), 3):
        A = params[i]
        mu = params[i + 1]
        sigma = params[i + 2]
        y += A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return y

def calculate_aicc(y_data, y_fit, num_params):
    """Corrected Akaike Information Criterion (AICc) balancing goodness-of-fit and simplicity."""
    n = len(y_data)
    residuals = y_data - y_fit
    sse = np.sum(residuals ** 2)
    if sse <= 0 or n <= num_params + 1:
        return np.inf
    aic = n * np.log(sse / n) + 2 * num_params
    correction = (2 * num_params * (num_params + 1)) / (n - num_params - 1)
    return aic + correction

def fit_n_peaks_with_linear_bg(x, y, n_peaks, prominence_ratio=0.05):
    """Fit N Gaussians + linear background with dynamic peak detection & bounds."""
    x_min, x_max = float(np.min(x)), float(np.max(x))
    x_span = max(x_max - x_min, 1.0)
    y_min, y_max = float(np.min(y)), float(np.max(y))
    y_span = max(y_max - y_min, 1.0)

    # Estimate initial linear baseline from endpoints (first 5% and last 5% of data)
    edge_n = max(1, len(x) // 20)
    x_left, y_left = np.mean(x[:edge_n]), np.mean(y[:edge_n])
    x_right, y_right = np.mean(x[-edge_n:]), np.mean(y[-edge_n:])
    a_init = (y_right - y_left) / max(x_right - x_left, 1e-6)
    b_init = y_left - a_init * x_left

    y_detrended = np.maximum(y - (a_init * x + b_init), 0)

    # Detect initial peak centers dynamically
    prominence = (np.max(y_detrended) - np.min(y_detrended)) * prominence_ratio
    peaks, _ = find_peaks(
        y_detrended,
        prominence=prominence,
        distance=max(1, len(x) // (n_peaks * 3))
    )
    if len(peaks) > 0:
        peaks = peaks[np.argsort(y_detrended[peaks])[::-1]]

    p0 = [a_init, b_init]
    max_slope = y_span / x_span
    lower_bounds = [-max_slope, y_min - y_span * 0.5]
    upper_bounds = [max_slope, y_max + y_span * 0.5]

    for k in range(n_peaks):
        if k < len(peaks):
            mu_init = float(x[peaks[k]])
            A_init = float(y_detrended[peaks[k]])
        elif n_peaks == 3 and k == 2 and len(peaks) >= 2:
            # For 3-peak bridging mode, place the 3rd peak between the top 2 peaks
            mu_init = 0.5 * (float(x[peaks[0]]) + float(x[peaks[1]]))
            A_init = float(np.max(y_detrended)) * 0.25
        else:
            mu_init = x_min + (k + 1) * x_span / (n_peaks + 1)
            A_init = float(np.max(y_detrended)) * 0.4 / (k + 1)

        sigma_init = x_span / (n_peaks * 5.0) if not (n_peaks == 3 and k == 2) else x_span / 6.0
        p0.extend([max(A_init, 1e-3), mu_init, max(sigma_init, 0.5)])

        lower_bounds.extend([0.0, x_min, 0.2])
        upper_bounds.extend([y_span * 1.5, x_max, x_span / 2.0])

    # Clip p0 strictly inside bounds to avoid ValueError
    p0 = np.clip(p0, np.array(lower_bounds) + 1e-6, np.array(upper_bounds) - 1e-6)

    popt, pcov = curve_fit(
        multi_gaussian_linear_bg,
        x,
        y,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=25000,
    )
    perr = np.sqrt(np.diag(pcov))

    y_pred = multi_gaussian_linear_bg(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    aicc = calculate_aicc(y, y_pred, len(popt))

    peaks_out = []
    for k in range(n_peaks):
        idx = 2 + k * 3
        A, mu, sigma = popt[idx], popt[idx + 1], popt[idx + 2]
        A_err, mu_err, sigma_err = perr[idx], perr[idx + 1], perr[idx + 2]
        peaks_out.append({
            "A": A, "A_err": A_err,
            "mu": mu, "mu_err": mu_err,
            "sigma": sigma, "sigma_err": sigma_err,
            "FWHM": 2.35482 * sigma,
            "Area": A * sigma * np.sqrt(2 * np.pi),
        })
    # Sort peaks left-to-right by mu
    peaks_out.sort(key=lambda d: d["mu"])

    return {
        "n_peaks": n_peaks,
        "a": popt[0],
        "b": popt[1],
        "peaks": peaks_out,
        "r2": r2,
        "aicc": aicc,
        "y_pred": y_pred,
    }

def adaptive_fit_lane(x, y, mode="Adaptive (Auto 2 vs 3 Peaks)", fixed_n=3, aicc_delta=4.0, prominence_ratio=0.05):
    """Run either adaptive AICc selection (2 vs 3 peaks) or fixed N-peak fitting."""
    if mode.startswith("Fixed"):
        res = fit_n_peaks_with_linear_bg(x, y, fixed_n, prominence_ratio)
        res["selection_note"] = f"Fixed {fixed_n}-Peak Model"
        return res

    # Adaptive mode: compare 2-peak vs 3-peak
    res2, res3 = None, None
    try:
        res2 = fit_n_peaks_with_linear_bg(x, y, 2, prominence_ratio)
    except Exception:
        pass
    try:
        res3 = fit_n_peaks_with_linear_bg(x, y, 3, prominence_ratio)
    except Exception:
        pass

    if res2 is None and res3 is None:
        raise RuntimeError("Both 2-peak and 3-peak nonlinear fits failed to converge.")
    if res2 is None:
        res3["selection_note"] = "3-Peak (2-Peak failed)"
        return res3
    if res3 is None:
        res2["selection_note"] = "2-Peak (3-Peak failed)"
        return res2

    if res3["aicc"] < res2["aicc"] - aicc_delta:
        res3["selection_note"] = f"3-Peak Bridging (ΔAICc = {res3['aicc'] - res2['aicc']:.1f})"
        return res3
    else:
        res2["selection_note"] = f"2-Peak Stable (ΔAICc = {res3['aicc'] - res2['aicc']:.1f})"
        return res2

def plot_fit(x, y, fit_res, title):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(x, y, "o", markersize=3, color="#64748b", alpha=0.55, label="Raw Profile")
    ax.plot(x, fit_res["y_pred"], "-", linewidth=2, color="#dc2626",
            label=f"Total Fit ($R^2$={fit_res['r2']:.4f})")

    baseline = fit_res["a"] * x + fit_res["b"]
    ax.plot(x, baseline, ":", color="#475569", linewidth=1.2, label="Linear Baseline (ax+b)")

    colors = plt.cm.tab10.colors
    for idx, p in enumerate(fit_res["peaks"], start=1):
        comp = baseline + p["A"] * np.exp(-0.5 * ((x - p["mu"]) / p["sigma"]) ** 2)
        c = colors[(idx - 1) % len(colors)]
        ax.plot(x, comp, "--", linewidth=1.5, color=c,
                label=f"Peak {idx}: A={p['A']:.1f}, μ={p['mu']:.1f}, σ={p['sigma']:.2f}")
        ax.fill_between(x, baseline, comp, color=c, alpha=0.12)

    ax.set_title(f"{title}\n[{fit_res['selection_note']}]", fontsize=10, fontweight="bold")
    ax.set_xlabel("Pixel / Distance")
    ax.set_ylabel("Intensity")
    ax.legend(fontsize=7.5, loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig

# ── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Model & Fitting Settings")
    fit_mode = st.radio(
        "Fitting Mode",
        options=["Adaptive (Auto 2 vs 3 Peaks)", "Fixed N Peaks"],
        index=0,
        help="Adaptive mode compares 2-peak and 3-peak models using AICc to detect intermediate bridging/smearing.",
    )
    if fit_mode == "Adaptive (Auto 2 vs 3 Peaks)":
        aicc_threshold = st.number_input(
            "AICc Improvement Threshold (ΔAICc)",
            min_value=0.0, max_value=50.0, value=4.0, step=1.0,
            help="3-peak model is chosen only if AICc_3 < AICc_2 - threshold."
        )
        fixed_n = 3
    else:
        fixed_n = st.number_input("Fixed Number of Peaks (N)", min_value=1, max_value=8, value=3, step=1)
        aicc_threshold = 4.0

    has_header = st.checkbox("CSV files contain header row", value=False)
    prominence_ratio = st.slider("Peak Detection Sensitivity", min_value=0.01, max_value=0.30, value=0.05, step=0.01)

# ── File Upload ──────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "📁 Upload ImageJ Lane Profile CSV/TXT files:",
    type=["csv", "txt"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("👆 Upload one or more CSV files (Column 1: X/Pixel, Column 2: Y/Intensity) to start.")
    st.stop()

# ── Run Analysis ─────────────────────────────────────────────────────────────
if st.button("🚀 Run Adaptive Multi-Gaussian Fit", type="primary"):
    results_rows = []
    figures = []

    progress = st.progress(0)
    for i, fobj in enumerate(uploaded_files):
        fobj.seek(0)
        try:
            df = pd.read_csv(fobj, header=0 if has_header else None)
            x = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().values
            y = pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna().values
            min_len = min(len(x), len(y))
            x, y = x[:min_len], y[:min_len]

            fit_res = adaptive_fit_lane(
                x, y,
                mode=fit_mode,
                fixed_n=int(fixed_n),
                aicc_delta=float(aicc_threshold),
                prominence_ratio=prominence_ratio,
            )
            fig = plot_fit(x, y, fit_res, title=fobj.name)
            figures.append(fig)

            m = re.search(r"(\d+(?:\.\d+)?)\s*%", fobj.name)
            conc_str = f"{m.group(1)}%" if m else "-"

            for peak_idx, p in enumerate(fit_res["peaks"], start=1):
                results_rows.append({
                    "Filename": fobj.name,
                    "Detected Conc.": conc_str,
                    "Selected Model": f"{fit_res['n_peaks']}-peak",
                    "AICc": round(fit_res["aicc"], 2),
                    "Peak #": f"Peak {peak_idx}",
                    "A (Amplitude)": round(p["A"], 3),
                    "A StdErr": round(p["A_err"], 3),
                    "mu (Center)": round(p["mu"], 3),
                    "mu StdErr": round(p["mu_err"], 3),
                    "sigma (Width)": round(p["sigma"], 3),
                    "sigma StdErr": round(p["sigma_err"], 3),
                    "FWHM (2.355σ)": round(p["FWHM"], 3),
                    "Area": round(p["Area"], 2),
                    "Bg Slope (a)": round(fit_res["a"], 5),
                    "Bg Intercept (b)": round(fit_res["b"], 3),
                    "R²": round(fit_res["r2"], 5),
                })
        except Exception as e:
            st.error(f"⚠️ Error processing `{fobj.name}`: {e}")

        progress.progress((i + 1) / len(uploaded_files))
    progress.empty()

    if results_rows:
        st.markdown("### 📊 Fitted Multi-Gaussian Parameters ($A$, $\mu$, $\sigma$)")
        res_df = pd.DataFrame(results_rows)
        st.dataframe(res_df, use_container_width=True)

        csv_buf = io.StringIO()
        res_df.to_csv(csv_buf, index=False)
        st.download_button(
            label="📥 Download Parameters CSV",
            data=csv_buf.getvalue(),
            file_name="adaptive_gaussian_parameters.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.markdown("### 📈 Individual Lane Deconvolution Plots")
        cols = st.columns(2)
        for idx, fig in enumerate(figures):
            with cols[idx % 2]:
                st.pyplot(fig)
                plt.close(fig)
