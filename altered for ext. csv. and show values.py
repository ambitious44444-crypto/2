import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

st.set_page_config(page_title="DNA Origami Peak Fitting & Parameter Extraction", layout="wide")

st.title("🧬 DNA Origami: Full Parameter Extraction & AIC Analysis")
st.write("Upload your experimental CSV files to extract amplitudes ($A$), centers ($\mu$), widths ($\sigma$), and AIC scores.")

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
    aic = n * np.log(rss / n) + 2 * k + (2 * k * (k + 1)) / (n - k - 1)
    return aic, rss

uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    uploaded_files = sorted(uploaded_files, key=lambda f: f.name)
    st.success(f"Successfully loaded and sorted {len(uploaded_files)} files.")
    
    results = []
    
    for file in uploaded_files:
        file.seek(0)
        df = pd.read_csv(file, header=None)
        
        x = pd.to_numeric(df[0], errors='coerce').values
        y = pd.to_numeric(df[1], errors='coerce').values
        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[valid_mask], y[valid_mask]
        
        if len(x) == 0:
            continue

        # Initial guesses
        m1 = (x >= 50) & (x <= 150)
        p1_x = x[m1][np.argmax(y[m1])] if np.any(m1) else 100.0
        p1_y = np.max(y[m1]) if np.any(m1) else 100.0
        
        m2 = (x >= 180) & (x <= 280)
        p2_x = x[m2][np.argmax(y[m2])] if np.any(m2) else 220.0
        p2_y = np.max(y[m2]) if np.any(m2) else 100.0

        bg_init = np.mean(y[:20]) if len(y) >= 20 else y[0]

        # --- 2-PEAK FITTING ---
        p0_2 = [-0.01, bg_init, p1_y, p1_x, 15.0, p2_y, p2_x, 20.0]
        bounds_2 = (
            [-1.0, -100.0, 0.0, 60.0, 1.0, 0.0, 180.0, 1.0],
            [1.0, 200.0, 1000.0, 150.0, 60.0, 1000.0, 280.0, 60.0]
        )
        
        aic2, popt2, perr2 = np.nan, None, None
        try:
            popt2, pcov2 = curve_fit(model_2peaks, x, y, p0=p0_2, bounds=bounds_2, maxfev=15000)
            aic2, _ = calc_aic(y, model_2peaks(x, *popt2), len(popt2))
            perr2 = np.sqrt(np.diagonal(pcov2))
        except:
            pass

        # --- 3-PEAK FITTING ---
        p0_3 = [-0.01, bg_init, p1_y, p1_x, 10.0, p2_y, p2_x, 18.0, p1_y * 0.3, p1_x - 15, 8.0]
        bounds_3 = (
            [-1.0, -100.0, 0.0, 70.0, 1.0, 0.0, 180.0, 1.0, 0.0, 40.0, 1.0],
            [1.0, 200.0, 1000.0, 150.0, 50.0, 1000.0, 280.0, 50.0, 200.0, 110.0, 30.0]
        )
        
        aic3, popt3, perr3 = np.nan, None, None
        try:
            popt3, pcov3 = curve_fit(model_3peaks, x, y, p0=p0_3, bounds=bounds_3, maxfev=15000)
            aic3, _ = calc_aic(y, model_3peaks(x, *popt3), len(popt3))
            perr3 = np.sqrt(np.diagonal(pcov3))
        except:
            pass

        # Model comparison
        selected_model = "Failed"
        if not np.isnan(aic2) and not np.isnan(aic3):
            selected_model = "3-Peak" if aic3 < aic2 - 2 else "2-Peak"
        elif not np.isnan(aic2):
            selected_model = "2-Peak"
        elif not np.isnan(aic3):
            selected_model = "3-Peak"

        row_data = {
            "Filename": file.name,
            "Winning Model": selected_model,
            "AIC (2-Pk)": round(aic2, 1) if not np.isnan(aic2) else "N/A",
            "AIC (3-Pk)": round(aic3, 1) if not np.isnan(aic3) else "N/A",
        }

        # Add 2-Peak parameters if successful
        if popt2 is not None:
            # popt2 indices: 0:bg_a, 1:bg_b, 2:A1, 3:mu1, 4:sig1, 5:A2, 6:mu2, 7:sig2
            row_data.update({
                "2P_A1": f"{popt2[2]:.1f} ± {perr2[2]:.1f}",
                "2P_μ1": f"{popt2[3]:.1f} ± {perr2[3]:.1f}",
                "2P_σ1": f"{popt2[4]:.1f} ± {perr2[4]:.1f}",
                "2P_A2": f"{popt2[5]:.1f} ± {perr2[5]:.1f}",
                "2P_μ2": f"{popt2[6]:.1f} ± {perr2[6]:.1f}",
                "2P_σ2": f"{popt2[7]:.1f} ± {perr2[7]:.1f}",
            })
        
        # Add 3-Peak parameters if successful
        if popt3 is not None:
            # popt3 indices: 0:bg_a, 1:bg_b, 2:A1, 3:mu1, 4:sig1, 5:A2, 6:mu2, 7:sig2, 8:A3, 9:mu3, 10:sig3
            row_data.update({
                "3P_A1": f"{popt3[2]:.1f} ± {perr3[2]:.1f}",
                "3P_μ1": f"{popt3[3]:.1f} ± {perr3[3]:.1f}",
                "3P_σ1": f"{popt3[4]:.1f} ± {perr3[4]:.1f}",
                "3P_A2": f"{popt3[5]:.1f} ± {perr3[5]:.1f}",
                "3P_μ2": f"{popt3[6]:.1f} ± {perr3[6]:.1f}",
                "3P_σ2": f"{popt3[7]:.1f} ± {perr3[7]:.1f}",
                "3P_A3": f"{popt3[8]:.1f} ± {perr3[8]:.1f}",
                "3P_μ3": f"{popt3[9]:.1f} ± {perr3[9]:.1f}",
                "3P_σ3": f"{popt3[10]:.1f} ± {perr3[10]:.1f}",
            })

        results.append(row_data)

    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True)
    
    # Optional: Download button for CSV export
    csv_data = df_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Parameter Results as CSV",
        data=csv_data,
        file_name="biomod_fitting_parameters.csv",
        mime="text/csv",
    )