import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib


MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

@st.cache_resource
def load_model():
    lin_mod = joblib.load(os.path.join(MODEL_DIR, "laptop_price_model.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "model_features.pkl"))
    return lin_mod, features

lin_reg_mod, model_features = load_model()


st.title("Laptop Price Prediction")
st.markdown("Fill in the laptop specifications below to predict its price.")

# ---------------------------
# SCREEN & DISPLAY
# ---------------------------
with st.expander("Screen & Display", expanded=True):
    inches = st.selectbox("Screen Size (inches)", [13.3, 14.0, 15.6, 16.0, 17.3])
    resolution = st.selectbox(
        "Resolution",
        [
            "1366x768",
            "1600x900",
            "1920x1080",
            "1920x1200",
            "2560x1440",
            "2560x1600",
            "2880x1800",
            "3000x2000",
            "3072x1920",
            "3456x2234",
            "3840x2160",
            "3840x2400",
        ],
    )
    touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
    weight = st.number_input("Weight (kg)", 0.8, 5.0, 2.0, 0.1)
    ram = st.selectbox("RAM (GB)", [4, 8, 12, 16, 24, 32, 64])

# Split resolution into width & height
screen_width, screen_height = map(int, resolution.split("x"))
ppi = ((screen_width**2 + screen_height**2) ** 0.5) / inches

# ---------------------------
# STORAGE
# ---------------------------
with st.expander("Storage"):
    hdd = st.selectbox("HDD (GB)", [0, 128, 256, 512, 1024])
    ssd = st.selectbox("SSD (GB)", [0, 128, 256, 512, 1024])
    flash = st.selectbox("Flash Storage (GB)", [0, 2, 4, 8, 16, 64])
    total_storage = hdd + ssd + flash

# ---------------------------
# CPU
# ---------------------------
with st.expander("CPU"):
    cpu_brand = st.selectbox("CPU Brand", ["Intel", "Samsung", "AMD"])
    if cpu_brand == "Intel":
        cpu_model = st.selectbox(
            "CPU Model",
            [
                "i3-10110U 2.1GHz",
                "i5-1035G1 1.0GHz",
                "i5-1135G7 2.4GHz",
                "i7-1065G7 1.3GHz",
                "i7-1165G7 2.8GHz",
                "i7-12700H 2.3GHz",
                "i9-10980HK 2.4GHz",
                "i9-13900HX 2.2GHz",
            ],
        )
    elif cpu_brand == "Samsung":
        cpu_model = st.selectbox(
            "CPU Model",
            ["Snapdragon 8cx Gen 3 3.0GHz", "Exynos 2200 2.8GHz"],
        )
    else:
        cpu_model = st.selectbox(
            "CPU Model",
            [
                "Ryzen 3 4300U 2.7GHz",
                "Ryzen 5 4500U 2.3GHz",
                "Ryzen 5 5600U 2.1GHz",
                "Ryzen 7 4700U 2.0GHz",
                "Ryzen 7 5800H 3.2GHz",
                "Ryzen 9 4900HS 3.0GHz",
                "Ryzen 9 5900HX 3.3GHz",
            ],
        )

cpu_speed = float(cpu_model.split()[-1].replace("GHz", ""))

# ---------------------------
# GPU
# ---------------------------
with st.expander("GPU"):
    gpu_spec = st.selectbox(
        "GPU",
        [
            "Intel Integrated",
            "Nvidia GTX 1650",
            "Nvidia RTX 3060",
            "AMD Radeon RX 5600M",
            "ARM Mali G76",
        ],
    )
    dedicated_gpu = 0 if "Integrated" in gpu_spec else 1

# Map GPU brand
if "Intel" in gpu_spec:
    gpu_brand = "Intel"
elif "Nvidia" in gpu_spec:
    gpu_brand = "Nvidia"
elif "AMD" in gpu_spec:
    gpu_brand = "AMD"
elif "ARM" in gpu_spec:
    gpu_brand = "ARM"
else:
    gpu_brand = "Other"

# ---------------------------
# COMPANY, TYPE & OS
# ---------------------------
with st.expander("Other Features"):
    company_selected = st.selectbox(
        "Company",
        [
            "Apple",
            "Asus",
            "Dell",
            "HP",
            "Lenovo",
            "MSI",
            "Razer",
            "Samsung",
            "Microsoft",
            "Xiaomi",
        ],
    )
    type_selected = st.selectbox(
        "Type", ["Gaming", "Netbook", "Notebook", "Ultrabook", "Workstation"]
    )
    os_selected = st.selectbox(
        "Operating System", ["Windows", "MacOS", "Linux", "Chrome OS", "No OS"]
    )

# ---------------------------
# BUILD INPUT DataFrame
# ---------------------------
numerical_data = {
    "Inches": inches,
    "Ram (GB)": ram,
    "Weight (kg)": weight,
    "HDD (GB)": hdd,
    "SSD (GB)": ssd,
    "Flash Storage (GB)": flash,
    "Total_Storage (GB)": total_storage,
    "Screen_Width": screen_width,
    "Screen_Height": screen_height,
    "PPI": ppi,
    "Touchscreen": 1 if touchscreen == "Yes" else 0,
    "Cpu_Speed_GHz": cpu_speed,
    "Gpu_Dedicated": dedicated_gpu,
    "Is_SSD": 1 if ssd > 0 else 0,
    "Is_High_RAM": 1 if ram >= 16 else 0,
}


# One-hot encoding helper
def one_hot(columns, selected_value, prefix):
    return {f"{prefix}_{col}": int(col == selected_value) for col in columns}


company_columns = [
    "Apple",
    "Asus",
    "Dell",
    "Fujitsu",
    "Google",
    "HP",
    "Huawei",
    "LG",
    "Lenovo",
    "MSI",
    "Mediacom",
    "Microsoft",
    "Razer",
    "Samsung",
    "Toshiba",
    "Vero",
    "Xiaomi",
]
type_columns = ["Gaming", "Netbook", "Notebook", "Ultrabook", "Workstation"]
os_columns = [
    "Clean_Windows",
    "Clean_MacOS",
    "Clean_Linux",
    "Clean_Chrome OS",
    "Clean_No OS",
]
cpu_columns = ["Intel", "Samsung"]
gpu_columns = ["ARM", "Intel", "Nvidia"]

categorical_data = {}
categorical_data.update(one_hot(company_columns, company_selected, "Company"))
categorical_data.update(one_hot(type_columns, type_selected, "TypeName"))
categorical_data.update(one_hot(os_columns, f"Clean_{os_selected}", "OpSys"))
categorical_data.update(one_hot(cpu_columns, cpu_brand, "Cpu_Brand"))
categorical_data.update(one_hot(gpu_columns, gpu_brand, "Gpu_Brand"))

# Combine all features
input_df = pd.DataFrame([{**numerical_data, **categorical_data}])

# Align with model features
for f in model_features:
    if f not in input_df.columns:
        input_df[f] = 0
input_df = input_df[model_features]

# for debugging
# st.write("Input features prepared for prediction:")
# st.dataframe(input_df)


# --- -- -- -- --
# Predict Button
# ----- -- -- --
if st.button("Predict Price"):
    # st.write("### Model Input Features", input_df)
    prediction = lin_reg_mod.predict(input_df)[0]
    if prediction < 0:
        st.warning(
            "Predicted price is negative. This may indicate the input combination is unrealistic or outside the model's training data. Setting price to $0."
        )
        prediction = 0.0
    st.success(f"Predicted Price: ${prediction:,.2f}")
