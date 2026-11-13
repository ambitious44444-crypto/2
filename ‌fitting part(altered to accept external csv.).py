import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

st.set_page_config(page_title="DNA Origami Peak Fitting & AIC Model Selection", layout="wide")

st.title("🧬 DNA Origami Degradation: 2-Peak vs. 3-Peak AIC Analysis")
st.write("Upload your experimental CSV files to run automated model selection using small-sample AIC correction.")

def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def model_2peaks(x, bg_a, bg_b, A1, mu1, sig1, A2, mu2, sig2):
    return bg_a * x + bg_b + gaussian(x, A1, mu1, sig1) + gaussian(x, A2, mu2, sig2)

def model_3peaks(x, bg_a, bg_b, A1, mu1, sig1, A2, mu2, sig2, A3, mu3, sig3):
    return (bg_a * x + bg_b + 
            gaussian(x, A1, mu1, sig1) + 
            gaussian(x, A2, mu2, sig2) + 
            gaussian(x, A3, mu3, sig3))

def calc_aic(y_true, y_pred, k):
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    if rss == 0:
        return np.inf, 0.0
    # Small-sample corrected AIC (AICc) formula
    aic = n * np.log(rss / n) + 2 * k + (2 * k * (k + 1)) / (n - k - 1)
    return aic, rss

uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    # Ensure deterministic alphabetical sorting matching local glob behavior
    uploaded_files = sorted(uploaded_files, key=lambda f: f.name)
    st.success(f"Successfully loaded and sorted {len(uploaded_files)} files.")
    
    results = []
    
    for file in uploaded_files:
        file.seek(0)
        df = pd.read_csv(file, header=None)
        
        # Force numeric conversion and clean invalid rows
        x = pd.to_numeric(df[0], errors='coerce').values
        y = pd.to_numeric(df[1], errors='coerce').values
        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[valid_mask], y[valid_mask]
        
        if len(x) == 0:
            continue

        # Dynamic initial guesses for peaks
        m1 = (x >= 50) & (x <= 150)
        p1_x = x[m1][np.argmax(y[m1])] if np.any(m1) else 100.0
        p1_y = np.max(y[m1]) if np.any(m1) else 100.0
        
        m2 = (x >= 180) & (x <= 280)
        p2_x = x[m2][np.argmax(y[m2])] if np.any(m2) else 220.0
        p2_y = np.max(y[m2]) if np.any(m2) else 100.0

        bg_init = np.mean(y[:20]) if len(y) >= 20 else y[0]

        # --- 2-PEAK MODEL FITTING ---
        p0_2 = [-0.01, bg_init, p1_y, p1_x, 15.0, p2_y, p2_x, 20.0]
        bounds_2 = (
            [-1.0, -100.0, 0.0, 60.0, 1.0, 0.0, 180.0, 1.0],
            [1.0, 200.0, 1000.0, 150.0, 60.0, 1000.0, 280.0, 60.0]
        )
        
        aic2 = np.nan
        try:
            popt2, _ = curve_fit(model_2peaks, x, y, p0=p0_2, bounds=bounds_2, maxfev=15000)
            aic2, _ = calc_aic(y, model_2peaks(x, *popt2), len(popt2))
        except:
            pass

        # --- 3-PEAK MODEL FITTING (Includes shoulder peak) ---
        p0_3 = [-0.01, bg_init, p1_y, p1_x, 10.0, p2_y, p2_x, 18.0, p1_y * 0.3, p1_x - 15, 8.0]
        bounds_3 = (
            [-1.0, -100.0, 0.0, 70.0, 1.0, 0.0, 180.0, 1.0, 0.0, 40.0, 1.0],
            [1.0, 200.0, 1000.0, 150.0, 50.0, 1000.0, 280.0, 50.0, 200.0, 110.0, 30.0]
        )
        
        aic3 = np.nan
        try:
            popt3, _ = curve_fit(model_3peaks, x, y, p0=p0_3, bounds=bounds_3, maxfev=15000)
            aic3, _ = calc_aic(y, model_3peaks(x, *popt3), len(popt3))
        except:
            pass

        # Model comparison via AIC (lower AIC indicates preferred model)
        selected_model = "Failed Fit"
        delta_aic = np.nan
        if not np.isnan(aic2) and not np.isnan(aic3):
            delta_aic = abs(aic2 - aic3)
            # Standard criterion: lower AIC wins. If delta is small (< 2), 2-peak is favored for parsimony.
            selected_model = "3-Peak" if aic3 < aic2 - 2 else "2-Peak"
        elif not np.isnan(aic2):
            selected_model = "2-Peak"
        elif not np.isnan(aic3):
            selected_model = "3-Peak"

        results.append({
            "Filename": file.name,
            "AIC (2-Peak)": round(aic2, 2) if not np.isnan(aic2) else "N/A",
            "AIC (3-Peak)": round(aic3, 2) if not np.isnan(aic3) else "N/A",
            "Winning Model": selected_model
        })

    # Display interactive results table
    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True)