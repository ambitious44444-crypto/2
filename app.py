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
    page_title="Multi-Gaussian Gel Profile Fitter",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Multi-Gaussian Nonlinear Fitter")
st.markdown(
    "Upload ImageJ lane profile CSV files (2 columns: `pixel`, `intensity`) to perform "
    "direct multi-Gaussian nonlinear least squares fitting and extract **$A$ (Amplitude)**, "
    "**$\mu$ (Center)**, and **$\sigma$ (Width)**."
)

# ── Helper Functions ─────────────────────────────────────────────────────────
def multi_gaussian(x, baseline, *params):
    """Sum of N Gaussian peaks plus a constant baseline."""
    y = np.full_like(x, baseline, dtype=float)
    for i in range(0, len(params), 3):
        A = params[i]
        mu = params[i + 1]
        sigma = params[i + 2]
        y += A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return y

def fit_multi_gaussian(x, y, n_peaks, min_prominence_ratio=0.05):
    """Fit N Gaussians with automatic initial peak detection and bounds."""
    baseline_init = float(np.percentile(y, 10))
    y_corr = np.maximum(y - baseline_init, 0)
    
    # Detect initial peak centers
    prominence = (np.max(y_corr) - np.min(y_corr)) * min_prominence_ratio
    peaks, props = find_peaks(y_corr, prominence=prominence, distance=max(1, len(x) // (n_peaks * 3)))
    
    # Sort detected peaks by height descending
    if len(peaks) > 0:
        sorted_idx = np.argsort(y_corr[peaks])[::-1]
        peaks = peaks[sorted_idx]

    # Build initial guess [baseline, A1, mu1, sigma1, ...]
    p0 = [baseline_init]
    lower_bounds = [0.0]
    upper_bounds = [float(np.max(y))]

    x_min, x_max = float(np.min(x)), float(np.max(x))
    x_span = max(x_max - x_min, 1.0)
    y_max = max(float(np.max(y)), 1.0)

    for k in range(n_peaks):
        if k < len(peaks):
            mu_init = float(x[peaks[k]])
            A_init = float(y_corr[peaks[k]])
        else:
            # Fallback evenly spaced centers if fewer peaks detected
            mu_init = x_min + (k + 1) * x_span / (n_peaks + 1)
            A_init = float(np.max(y_corr)) * 0.5 / (k + 1)
        
        sigma_init = x_span / (n_peaks * 6.0)
        p0.extend([max(A_init, 1e-3), mu_init, max(sigma_init, 0.5)])
        
        lower_bounds.extend([0.0, x_min, 0.1])
        upper_bounds.extend([y_max * 1.5, x_max, x_span / 2.0])

    popt, pcov = curve_fit(
        multi_gaussian,
        x,
        y,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=20000,
    )
    perr = np.sqrt(np.diag(pcov))
    
    # Calculate R-squared
    y_pred = multi_gaussian(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Extract peak components sorted by mu (left to right on gel)
    peaks_out = []
    for k in range(n_peaks):
        idx = 1 + k * 3
        A, mu, sigma = popt[idx], popt[idx + 1], popt[idx + 2]
        A_err, mu_err, sigma_err = perr[idx], perr[idx + 1], perr[idx + 2]
        fwhm = 2.35482 * sigma
        area = A * sigma * np.sqrt(2 * np.pi)
        peaks_out.append({
            "A": A, "A_err": A_err,
            "mu": mu, "mu_err": mu_err,
            "sigma": sigma, "sigma_err": sigma_err,
            "FWHM": fwhm, "Area": area
        })
    peaks_out.sort(key=lambda d: d["mu"])

    return {
        "baseline": popt[0],
        "peaks": peaks_out,
        "r2": r2,
        "y_pred": y_pred,
    }

def plot_fit(x, y, fit_res, title):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(x, y, "o", markersize=3, color="#64748b", alpha=0.6, label="Raw Data")
    ax.plot(x, fit_res["y_pred"], "-", linewidth=2, color="#dc2626", label=f"Total Fit ($R^2$={fit_res['r2']:.4f})")
    
    b = fit_res["baseline"]
    ax.axhline(b, linestyle=":", color="#94a3b8", linewidth=1, label="Baseline")
    
    colors = plt.cm.tab10.colors
    for idx, p in enumerate(fit_res["peaks"], start=1):
        comp = b + p["A"] * np.exp(-0.5 * ((x - p["mu"]) / p["sigma"]) ** 2)
        c = colors[(idx - 1) % len(colors)]
        ax.plot(x, comp, "--", linewidth=1.5, color=c, label=f"Peak {idx} (μ={p['mu']:.1f}, σ={p['sigma']:.2f})")
        ax.fill_between(x, b, comp, color=c, alpha=0.12)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Pixel / Distance")
    ax.set_ylabel("Intensity")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig

# ── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Fitting Settings")
    n_peaks = st.number_input("Number of Gaussian Peaks (N)", min_value=1, max_value=8, value=3, step=1)
    has_header = st.checkbox("CSV files contain header row", value=False)
    prominence_ratio = st.slider("Peak Detection Sensitivity", min_value=0.01, max_value=0.30, value=0.05, step=0.01)

# ── File Upload ──────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "📁 Upload ImageJ Lane Profile CSV/TXT files:",
    type=["csv", "txt"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("👆 Please upload one or more CSV files (Column 1: X/Pixel, Column 2: Y/Intensity) to begin.")
    st.stop()

# ── Run Fitting ──────────────────────────────────────────────────────────────
if st.button("🚀 Run Multi-Gaussian Fit", type="primary"):
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

            fit_res = fit_multi_gaussian(x, y, int(n_peaks), prominence_ratio)
            fig = plot_fit(x, y, fit_res, title=f"{fobj.name} (N={n_peaks})")
            figures.append(fig)

            # Extract concentration from filename if available
            m = re.search(r"(\d+(?:\.\d+)?)\s*%", fobj.name)
            conc_str = f"{m.group(1)}%" if m else "-"

            for peak_idx, p in enumerate(fit_res["peaks"], start=1):
                results_rows.append({
                    "Filename": fobj.name,
                    "Detected Conc.": conc_str,
                    "Peak #": f"Peak {peak_idx}",
                    "A (Amplitude)": round(p["A"], 3),
                    "A StdErr": round(p["A_err"], 3),
                    "mu (Center)": round(p["mu"], 3),
                    "mu StdErr": round(p["mu_err"], 3),
                    "sigma (Width)": round(p["sigma"], 3),
                    "sigma StdErr": round(p["sigma_err"], 3),
                    "FWHM (2.355σ)": round(p["FWHM"], 3),
                    "Gaussian Area": round(p["Area"], 2),
                    "Baseline": round(fit_res["baseline"], 3),
                    "R²": round(fit_res["r2"], 5),
                })
        except Exception as e:
            st.error(f"⚠️ Failed to process `{fobj.name}`: {e}")
        
        progress.progress((i + 1) / len(uploaded_files))
    progress.empty()

    if results_rows:
        st.markdown("### 📊 Fitted Gaussian Parameters ($A$, $\mu$, $\sigma$)")
        res_df = pd.DataFrame(results_rows)
        st.dataframe(res_df, use_container_width=True)

        csv_buffer = io.StringIO()
        res_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Parameters Table (CSV)",
            data=csv_buffer.getvalue(),
            file_name="multi_gaussian_parameters.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.markdown("### 📈 Spectral Decomposition Plots")
        cols = st.columns(2)
        for idx, fig in enumerate(figures):
            with cols[idx % 2]:
                st.pyplot(fig)
                plt.close(fig)
