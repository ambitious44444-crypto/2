import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==========================================
# 頁面基本設定
# ==========================================
st.set_page_config(
    page_title="光譜擬合與相變閾值分析系統",
    page_icon="📈",
    layout="wide"
)

st.title("🔬 光譜多峰擬合與濃度飽和閾值分析儀")
st.markdown("這是一個專門用於處理多分散系統光譜、自動化模型選擇（AIC）與尋找巨觀相變閾值的分析工具。")

# ==========================================
# 數學模型定義
# ==========================================
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def model_2peaks(x, a, b, A1, mu1, sig1, A2, mu2, sig2):
    return a * x + b + gaussian(x, A1, mu1, sig1) + gaussian(x, A2, mu2, sig2)

def model_3peaks(x, a, b, A1, mu1, sig1, A2, mu2, sig2, A3, mu3, sig3):
    return a * x + b + gaussian(x, A1, mu1, sig1) + gaussian(x, A2, mu2, sig2) + gaussian(x, A3, mu3, sig3)

def calculate_aic(y_data, y_fit, num_params):
    n = len(y_data)
    residuals = y_data - y_fit
    sse = np.sum(residuals**2)
    if sse == 0:
        return np.inf
    return n * np.log(sse / n) + 2 * num_params + (2 * num_params * (num_params + 1)) / (n - num_params - 1)

def piecewise_linear(x, x_break, y_intercept, slope1, slope2):
    return y_intercept + slope1 * x + slope2 * np.maximum(0, x - x_break)

# ==========================================
# 側邊欄導航
# ==========================================
st.sidebar.header("⚙️ 功能選單")
app_mode = st.sidebar.selectbox(
    "選擇分析模組",
    ["1. 單張光譜 AIC 自動擬合與特徵萃取", "2. 濃度序列與閾值拐點 (Threshold) 分析"]
)

# ==========================================
# 模組一：單張光譜分析
# ==========================================
if app_mode == "1. 單張光譜 AIC 自動擬合與特徵萃取":
    st.header("📊 單張光譜資料的自動化模型選擇 (2峰 vs 3峰)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("資料輸入與設定")
        data_source = st.radio("選擇數據來源", ["使用內建模擬高濃度範例", "上傳自訂 CSV 檔案"])
        
        if data_source == "上傳自訂 CSV 檔案":
            uploaded_file = st.file_uploader("上傳 CSV (格式: x, y)", type=["csv", "txt"])
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file, header=None)
                x_data, y_data = df.iloc[:, 0].values, df.iloc[:, 1].values
            else:
                st.info("請上傳檔案，或切換至內建範例。")
                x_data, y_data = None, None
        else:
            # 內建高濃度模擬數據 (含有橋接態)
            np.random.seed(42)
            x_data = np.linspace(0, 100, 400)
            y_data = (0.001 * x_data + 0.05 + 
                      gaussian(x_data, 0.5, 30, 4.5) + 
                      gaussian(x_data, 0.6, 70, 5.0) + 
                      gaussian(x_data, 0.3, 50, 12.0) + 
                      np.random.normal(0, 0.01, 400))
            st.success("已載入內建高濃度模擬光譜（含橋接態）")

    if x_data is not None and y_data is not None:
        with col2:
            st.subheader("擬合結果與模型診斷")
            
            # 執行擬合
            guess_2 = [0.0, 0.05, 0.6, 30, 4.0, 0.7, 70, 4.0]
            guess_3 = [0.0, 0.05, 0.6, 30, 4.0, 0.7, 70, 4.0, 0.3, 50, 10.0]
            
            try:
                popt_2, _ = curve_fit(model_2peaks, x_data, y_data, p0=guess_2, maxfev=5000)
                y_fit_2 = model_2peaks(x_data, *popt_2)
                aic_2 = calculate_aic(y_data, y_fit_2, len(popt_2))
            except:
                aic_2, popt_2 = np.inf, None

            try:
                popt_3, _ = curve_fit(model_3peaks, x_data, y_data, p0=guess_3, maxfev=5000)
                y_fit_3 = model_3peaks(x_data, *popt_3)
                aic_3 = calculate_aic(y_data, y_fit_3, len(popt_3))
            except:
                aic_3, popt_3 = np.inf, None

            # AIC 決策
            if popt_3 is not None and (aic_3 < aic_2 - 4):
                selected_model = "3-peak"
                popt = popt_3
                y_fit = y_fit_3
                st.markdown(f"🔥 **AIC 診斷結果**：偵測到顯著的過渡態，自動採用 **3 峰模型** (AIC: {aic_3:.2f} vs 2峰 AIC: {aic_2:.2f})")
            else:
                selected_model = "2-peak"
                popt = popt_2
                y_fit = y_fit_2
                st.markdown(f"🛡️ **AIC 診斷結果**：系統結構穩定，自動採用 **2 峰模型** 以防過擬合 (AIC: {aic_2:.2f} vs 3峰 AIC: {aic_3:.2f})")

            # 繪圖
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(x_data, y_data, color='gray', s=10, alpha=0.6, label='Raw Data')
            ax.plot(x_data, y_fit, color='red', linewidth=2, label='Total Fit')
            
            if selected_model == "3-peak":
                bg = popt[0]*x_data + popt[1]
                ax.plot(x_data, bg + gaussian(x_data, popt[2], popt[3], popt[4]), '--', color='blue', label='State 1')
                ax.plot(x_data, bg + gaussian(x_data, popt[5], popt[6], popt[7]), '--', color='green', label='State 2 ($\sigma_2$ target)')
                ax.plot(x_data, bg + gaussian(x_data, popt[8], popt[9], popt[10]), '-.', color='purple', label='Bridge State 3')
                extracted_sigma2 = popt[7]
            else:
                bg = popt[0]*x_data + popt[1]
                ax.plot(x_data, bg + gaussian(x_data, popt[2], popt[3], popt[4]), '--', color='blue', label='State 1')
                ax.plot(x_data, bg + gaussian(x_data, popt[5], popt[6], popt[7]), '--', color='green', label='State 2 ($\sigma_2$ target)')
                extracted_sigma2 = popt[7]

            ax.set_xlabel("Reaction Coordinate / X")
            ax.set_ylabel("Intensity")
            ax.legend(frameon=False)
            st.pyplot(fig)
            
            st.metric(label="萃取出的目標態寬度 ($\sigma_2$)", value=f"{extracted_sigma2:.4f}")

# ==========================================
# 模組二：濃度序列與閾值分析
# ==========================================
elif app_mode == "2. 濃度序列與閾值拐點 (Threshold) 分析":
    st.header("📈 巨觀相變閾值計算 (Piecewise Linear Regression)")
    st.markdown("將不同濃度下萃取出的 $\sigma_2$ 數據進行區段線性擬合，客觀找出系統崩潰或飽和的轉折點。")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("參數調整與資料")
        use_default_series = st.checkbox("使用內建範例濃度序列", value=True)
        
        if use_default_series:
            conc_data = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
            sigma2_data = np.array([4.1, 4.12, 4.15, 4.18, 4.25, 4.5, 5.8, 7.5, 9.1, 10.2, 10.8])
            st.info("已載入標準範例數據（閾值約在 1.0 左右）")
        else:
            uploaded_csv = st.file_uploader("上傳包含 Concentration 與 Sigma2 的 CSV", type=["csv"])
            if uploaded_csv is not None:
                df_series = pd.read_csv(uploaded_csv)
                conc_data = df_series['Concentration'].values
                sigma2_data = df_series['Sigma2'].values
            else:
                conc_data, sigma2_data = None, None

    if conc_data is not None:
        with col2:
            st.subheader("區段線性擬合結果")
            
            # 執行區段線性擬合
            initial_guess = [1.0, 4.1, 0.1, 5.0]
            bounds = ([0.0, 3.0, -1.0, 0.0], [3.0, 5.0, 2.0, 30.0])
            
            try:
                popt_pw, pcov_pw = curve_fit(piecewise_linear, conc_data, sigma2_data, p0=initial_guess, bounds=bounds)
                threshold = popt_pw[0]
                perr = np.sqrt(np.diag(pcov_pw))
                threshold_err = perr[0]
                
                # 繪製圖表
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                ax2.scatter(conc_data, sigma2_data, color='blue', s=30, label='Extracted $\sigma_2$ Data')
                ax2.plot(conc_data, piecewise_linear(conc_data, *popt_pw), color='red', linewidth=2.5, label='Piecewise Fit')
                ax2.axvline(x=threshold, color='gray', linestyle='--', label=f'Threshold = {threshold:.3f} (±{threshold_err:.3f})')
                
                ax2.set_xlabel("Concentration (C)", fontsize=12)
                ax2.set_ylabel("State 2 Width ($\sigma_2$)", fontsize=12)
                ax2.legend(frameon=False)
                ax2.grid(True, linestyle=':', alpha=0.6)
                st.pyplot(fig2)
                
                st.success(f"🎯 **成功計算出系統飽和閾值 (Threshold)**: **{threshold:.4f}** (標準差: ±{threshold_err:.4f})")
                
            except Exception as e:
                st.error(f"擬合過程發生錯誤，請調整初始猜測值或檢查數據格式：{e}")