from pyexpat import model
import numpy as np
import pandas as pd
import streamlit as st
import joblib


lin_reg_mod = joblib.load("../models/laptop_price_model.pkl")
model_features = joblib.load("../models/model_features.pkl")


# # Dynamically extract categories for selectboxes from model_features
# def extract_categories(prefix):
#     return sorted(
#         {
#             col.replace(prefix + "_", "")
#             for col in lin_reg_mod_features
#             if col.startswith(prefix + "_")
#         }
#     )


# cpu_options = extract_categories("CPU")
# gpu_options = extract_categories("GPU")
# os_options = extract_categories("OS")
# storage_options = extract_categories("Storage_Type")
# touchscreen_options = extract_categories("Touchscreen")

# st.title("Laptop Price Prediction")


# # --- Basic Specs ---
# with st.expander("Basic Specs"):
#     col1, col2 = st.columns(2)
#     with col1:
#         ram = st.slider("RAM (GB)", min_value=4, max_value=64, step=4, value=8)
#         storage = st.selectbox("Storage Type", ["HDD", "SSD", "Hybrid"], index=1)
#     with col2:
#         storage_size = st.number_input("Storage Size (GB)", min_value=128, max_value=4096, step=128, value=512)
#         weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, step=0.1, value=2.0)

# # --- Processor & GPU ---
# with st.expander("Processor & GPU"):
#     cpu = st.selectbox("CPU", ["Intel i3", "Intel i5", "Intel i7", "AMD Ryzen 3", "AMD Ryzen 5", "AMD Ryzen 7"])
#     gpu = st.selectbox("GPU", ["Integrated", "NVIDIA GTX", "NVIDIA RTX", "AMD Radeon"])

# # --- Display & Battery ---
# with st.expander("Display & Battery"):
#     screen_size = st.slider("Screen Size (inch)", min_value=10.0, max_value=20.0, step=0.1, value=15.6)
#     battery = st.number_input("Battery Capacity (Wh)", min_value=20, max_value=100, step=1, value=50)

# # --- Connectivity & Others ---
# with st.expander("Connectivity & Others"):
#     os = st.selectbox("Operating System", ["Windows", "Linux", "MacOS", "Other"])
#     touchscreen = st.selectbox("Touchscreen", ["Yes", "No"])


# with st.expander("Basic Specs"):
#     ram = st.slider("RAM (GB)", 4, 64, 8, 4)
#     storage = st.selectbox("Storage Type", storage_options)
#     storage_size = st.number_input("Storage Size (GB)", 128, 4096, 512, 128)
#     weight = st.number_input("Weight (kg)", 0.5, 5.0, 2.0, 0.1)

# with st.expander("Processor & GPU"):
#     cpu = st.selectbox("CPU", cpu_options)
#     gpu = st.selectbox("GPU", gpu_options)

# with st.expander("Display & Battery"):
#     screen_size = st.slider("Screen Size (inch)", 10.0, 20.0, 15.6, 0.1)
#     # battery = st.number_input("Battery Capacity (Wh)", 20, 100, 50, 1)

# with st.expander("Connectivity & Others"):
#     os = st.selectbox("Operating System", os_options)
#     touchscreen = st.selectbox("Touchscreen", touchscreen_options)

# # --- Prepare Input for Model ---
# input_dict = {
#     "RAM": ram,
#     "Storage_Type": storage,
#     "Storage_Size": storage_size,
#     "Weight": weight,
#     "CPU": cpu,
#     "GPU": gpu,
#     "Screen_Size": screen_size,
#     # "Battery": battery,
#     "OS": os,
#     "Touchscreen": touchscreen,
# }

# # Convert to DataFrame and align features
# input_df = pd.DataFrame([input_dict])
# input_df = pd.get_dummies(input_df)
# # Make sure all model features are present
# for col in lin_reg_mod_features:
#     if col not in input_df.columns:
#         input_df[col] = 0
# input_df = input_df[lin_reg_mod_features]

# # --- Predict Button ---
# if st.button("Predict Price"):
#     st.write("### Model Input Features", input_df)
#     prediction = lin_reg_mod.predict(input_df)[0]
#     if prediction < 0:
#         st.warning(
#             "Predicted price is negative. This may indicate the input combination is unrealistic or outside the model's training data. Setting price to $0."
#         )
#         prediction = 0.0
#     st.success(f"Predicted Price: ${prediction:,.2f}")


# st.title("Laptop Price Prediction")

# # ---------------------------
# # NUMERICAL INPUTS
# # ---------------------------
# inches = st.number_input("Screen Size (inches)", min_value=10.0, max_value=20.0, step=0.1, value=15.6)
# ram = st.number_input("RAM (GB)", min_value=2, max_value=64, step=2, value=8)
# weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, step=0.1, value=2.0)
# hdd = st.number_input("HDD (GB)", min_value=0, max_value=2000, step=128, value=0)
# ssd = st.number_input("SSD (GB)", min_value=0, max_value=2000, step=128, value=256)
# hybrid = st.number_input("Hybrid Storage (GB)", min_value=0, max_value=2000, step=128, value=0)
# flash = st.number_input("Flash Storage (GB)", min_value=0, max_value=2000, step=128, value=0)
# total_storage = st.number_input("Total Storage (GB)", min_value=0, max_value=4000, step=128, value=256)
# screen_width = st.number_input("Screen Width (px)", min_value=800, max_value=5000, step=10, value=1920)
# screen_height = st.number_input("Screen Height (px)", min_value=600, max_value=5000, step=10, value=1080)
# touchscreen = st.selectbox("Touchscreen", [0, 1])
# cpu_speed = st.number_input("CPU Speed (GHz)", min_value=1.0, max_value=5.0, step=0.1, value=2.5)

# # ---------------------------
# # CATEGORICAL INPUTS (ONE-HOT)
# # ---------------------------
# company_selected = st.selectbox("Company", [
#     "Apple", "Asus", "Chuwi", "Dell", "Fujitsu", "Google", "HP",
#     "Huawei", "LG", "Lenovo", "MSI", "Mediacom", "Microsoft", "Razer",
#     "Samsung", "Toshiba", "Vero", "Xiaomi"
# ])

# type_selected = st.selectbox("Type", ["Gaming", "Netbook", "Notebook", "Ultrabook", "Workstation"])
# os_selected = st.selectbox("Operating System", ["Clean_Chrome OS", "Clean_Linux", "Clean_MacOS", "Clean_No OS", "Clean_Windows"])
# cpu_brand_selected = st.selectbox("CPU Brand", ["Intel", "Samsung"])
# gpu_brand_selected = st.selectbox("GPU Brand", ["ARM", "Intel", "Nvidia"])

# # ---------------------------
# # BUILD INPUT DF
# # ---------------------------
# # Numerical features
# numerical_data = {
#     "Inches": inches,
#     "Ram (GB)": ram,
#     "Weight (kg)": weight,
#     "HDD (GB)": hdd,
#     "SSD (GB)": ssd,
#     "Hybrid (GB)": hybrid,
#     "Flash Storage (GB)": flash,
#     "Total_Storage (GB)": total_storage,
#     "Screen_Width": screen_width,
#     "Screen_Height": screen_height,
#     "Touchscreen": touchscreen,
#     "Cpu_Speed_GHz": cpu_speed
# }

# # One-hot features
# def one_hot(columns, selected_value, prefix):
#     return {f"{prefix}_{col}": int(col == selected_value) for col in columns}

# company_columns = ["Apple", "Asus", "Chuwi", "Dell", "Fujitsu", "Google", "HP",
#                    "Huawei", "LG", "Lenovo", "MSI", "Mediacom", "Microsoft", "Razer",
#                    "Samsung", "Toshiba", "Vero", "Xiaomi"]
# type_columns = ["Gaming", "Netbook", "Notebook", "Ultrabook", "Workstation"]
# os_columns = ["Clean_Chrome OS", "Clean_Linux", "Clean_MacOS", "Clean_No OS", "Clean_Windows"]
# cpu_brand_columns = ["Intel", "Samsung"]
# gpu_brand_columns = ["ARM", "Intel", "Nvidia"]

# categorical_data = {}
# categorical_data.update(one_hot(company_columns, company_selected, "Company"))
# categorical_data.update(one_hot(type_columns, type_selected, "TypeName"))
# categorical_data.update(one_hot(os_columns, os_selected, "OpSys"))
# categorical_data.update(one_hot(cpu_brand_columns, cpu_brand_selected, "Cpu_Brand"))
# categorical_data.update(one_hot(gpu_brand_columns, gpu_brand_selected, "Gpu_Brand"))

# # Combine all features
# input_df = pd.DataFrame([{**numerical_data, **categorical_data}])

# # Align with model features
# for f in model_features:
#     if f not in input_df.columns:
#         input_df[f] = 0
# input_df = input_df[model_features]


# st.title("💻 Laptop Price Prediction")

# st.markdown("Fill in the laptop specifications below to predict its price.")

# # ---------------------------
# # SCREEN AND DISPLAY
# # ---------------------------
# st.header("Screen & Display")
# col1, col2, col3 = st.columns(3)

# with col1:
#     inches = st.number_input("Screen Size (inches)", 10.0, 20.0, 15.6, 0.1)
#     screen_width = st.number_input("Screen Width (px)", 800, 5000, 1920, 10)
#     screen_height = st.number_input("Screen Height (px)", 600, 5000, 1080, 10)

# with col2:
#     touchscreen = st.selectbox("Touchscreen", [0, 1])
#     total_storage = st.number_input("Total Storage (GB)", 0, 4000, 256, 128)

# with col3:
#     cpu_speed = st.number_input("CPU Speed (GHz)", 1.0, 5.0, 2.5, 0.1)

# # ---------------------------
# # STORAGE & MEMORY
# # ---------------------------
# st.header("Storage & Memory")
# col1, col2, col3, col4 = st.columns(4)

# with col1:
#     ram = st.number_input("RAM (GB)", 2, 64, 8, 2)
# with col2:
#     hdd = st.number_input("HDD (GB)", 0, 2000, 0, 128)
# with col3:
#     ssd = st.number_input("SSD (GB)", 0, 2000, 256, 128)
# with col4:
#     hybrid = st.number_input("Hybrid Storage (GB)", 0, 2000, 0, 128)

# flash = st.number_input("Flash Storage (GB)", 0, 2000, 0, 128)

# # ---------------------------
# # PHYSICAL SPECS
# # ---------------------------
# st.header("Physical Specifications")
# weight = st.number_input("Weight (kg)", 0.5, 5.0, 2.0, 0.1)

# # ---------------------------
# # CATEGORICAL INPUTS
# # ---------------------------
# st.header("Categorical Features")

# col1, col2 = st.columns(2)

# with col1:
#     company_selected = st.selectbox("Company", [
#         "Apple","Asus","Chuwi","Dell","Fujitsu","Google","HP","Huawei","LG",
#         "Lenovo","MSI","Mediacom","Microsoft","Razer","Samsung","Toshiba",
#         "Vero","Xiaomi"])
#     type_selected = st.selectbox("Type", ["Gaming","Netbook","Notebook","Ultrabook","Workstation"])
#     os_selected = st.selectbox("Operating System", ["Clean_Chrome OS","Clean_Linux","Clean_MacOS","Clean_No OS","Clean_Windows"])

# with col2:
#     cpu_brand_selected = st.selectbox("CPU Brand", ["Intel","Samsung"])
#     gpu_brand_selected = st.selectbox("GPU Brand", ["ARM","Intel","Nvidia"])

# # ---------------------------
# # BUILD INPUT DF
# # ---------------------------
# # Numerical features
# numerical_data = {
#     "Inches": inches,
#     "Ram (GB)": ram,
#     "Weight (kg)": weight,
#     "HDD (GB)": hdd,
#     "SSD (GB)": ssd,
#     "Hybrid (GB)": hybrid,
#     "Flash Storage (GB)": flash,
#     "Total_Storage (GB)": total_storage,
#     "Screen_Width": screen_width,
#     "Screen_Height": screen_height,
#     "Touchscreen": touchscreen,
#     "Cpu_Speed_GHz": cpu_speed
# }

# # One-hot features
# def one_hot(columns, selected_value, prefix):
#     return {f"{prefix}_{col}": int(col == selected_value) for col in columns}

# company_columns = ["Apple","Asus","Chuwi","Dell","Fujitsu","Google","HP","Huawei","LG","Lenovo",
#                    "MSI","Mediacom","Microsoft","Razer","Samsung","Toshiba","Vero","Xiaomi"]
# type_columns = ["Gaming","Netbook","Notebook","Ultrabook","Workstation"]
# os_columns = ["Clean_Chrome OS","Clean_Linux","Clean_MacOS","Clean_No OS","Clean_Windows"]
# cpu_brand_columns = ["Intel","Samsung"]
# gpu_brand_columns = ["ARM","Intel","Nvidia"]

# categorical_data = {}
# categorical_data.update(one_hot(company_columns, company_selected, "Company"))
# categorical_data.update(one_hot(type_columns, type_selected, "TypeName"))
# categorical_data.update(one_hot(os_columns, os_selected, "OpSys"))
# categorical_data.update(one_hot(cpu_brand_columns, cpu_brand_selected, "Cpu_Brand"))
# categorical_data.update(one_hot(gpu_brand_columns, gpu_brand_selected, "Gpu_Brand"))

# # Combine all features
# input_df = pd.DataFrame([{**numerical_data, **categorical_data}])

# # Align with model features
# for f in model_features:
#     if f not in input_df.columns:
#         input_df[f] = 0
# input_df = input_df[model_features]


st.title("💻 Laptop Price Prediction")
st.markdown("Fill in the laptop specifications below to predict its price.")

# ---------------------------
# SCREEN & DISPLAY
# ---------------------------
with st.expander("Screen & Display", expanded=True):
    inches = st.selectbox("Screen Size (inches)", [13.3, 14.0, 15.6, 16.0, 17.3])
    resolution = st.selectbox(
        "Resolution", ["1366x768", "1920x1080", "2560x1440", "3840x2160"]
    )
    touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
    weight = st.number_input("Weight (kg)", 0.8, 5.0, 2.0, 0.1)
    ram = st.selectbox("RAM (GB)", [4, 8, 12, 16, 24, 32, 64])

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
    cpu_brand = st.selectbox("CPU Brand", ["Intel", "AMD"])
    # Each CPU model here implies its speed
    if cpu_brand == "Intel":
        cpu_model = st.selectbox(
            "CPU Model",
            [
                "i3-10110U 2.1GHz",
                "i5-1035G1 1.0GHz",
                "i7-1065G7 1.3GHz",
                "i9-10980HK 2.4GHz",
            ],
        )
    else:
        cpu_model = st.selectbox(
            "CPU Model",
            [
                "Ryzen 3 4300U 2.7GHz",
                "Ryzen 5 4500U 2.3GHz",
                "Ryzen 7 4700U 2.0GHz",
                "Ryzen 9 4900HS 3.0GHz",
            ],
        )

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
        ],
    )
    dedicated_gpu = 0 if "Integrated" in gpu_spec else 1

# ---------------------------
# COMPANY, TYPE & OS
# ---------------------------
with st.expander("Other Features"):
    company_selected = st.selectbox(
        "Company",
        ["Apple", "Asus", "Dell", "HP", "Lenovo", "MSI", "Acer", "Razer", "Samsung"],
    )
    type_selected = st.selectbox(
        "Type", ["Gaming", "Netbook", "Notebook", "Ultrabook", "Workstation"]
    )
    os_selected = st.selectbox(
        "Operating System", ["Windows", "MacOS", "Linux", "Chrome OS", "No OS"]
    )

# ---------------------------
# BUILD INPUT DF
# ---------------------------
# Extract CPU speed from model string
cpu_speed = float(cpu_model.split()[-1].replace("GHz", ""))

numerical_data = {
    "Inches": inches,
    "Ram (GB)": ram,
    "Weight (kg)": weight,
    "HDD (GB)": hdd,
    "SSD (GB)": ssd,
    "Flash Storage (GB)": flash,
    "Total_Storage (GB)": total_storage,
    "Resolution": resolution,
    "Touchscreen": 1 if touchscreen == "Yes" else 0,
    "Cpu_Speed_GHz": cpu_speed,
    "Dedicated_GPU": dedicated_gpu,
}


# One-hot encoding helper
def one_hot(columns, selected_value, prefix):
    return {f"{prefix}_{col}": int(col == selected_value) for col in columns}


company_columns = [
    "Apple",
    "Asus",
    "Dell",
    "HP",
    "Lenovo",
    "MSI",
    "Acer",
    "Razer",
    "Samsung",
]
type_columns = ["Gaming", "Netbook", "Notebook", "Ultrabook", "Workstation"]
os_columns = ["Windows", "MacOS", "Linux", "Chrome OS", "No OS"]

categorical_data = {}
categorical_data.update(one_hot(company_columns, company_selected, "Company"))
categorical_data.update(one_hot(type_columns, type_selected, "TypeName"))
categorical_data.update(one_hot(os_columns, os_selected, "OpSys"))

# Combine all features
input_df = pd.DataFrame([{**numerical_data, **categorical_data}])

# Align with model features
for f in model_features:
    if f not in input_df.columns:
        input_df[f] = 0
input_df = input_df[model_features]

st.write("✅ Input features prepared for prediction:")
st.dataframe(input_df)


# ---------------------------
# PREDICTION
# ---------------------------
# if st.button("Predict Price"):
#     predicted_price = lin_reg_mod.predict(input_df)[0]
#     st.success(f"Predicted Price: ${predicted_price:.2f}")

# --- Predict Button ---
if st.button("Predict Price"):
    st.write("### Model Input Features", input_df)
    prediction = lin_reg_mod.predict(input_df)[0]
    if prediction < 0:
        st.warning(
            "Predicted price is negative. This may indicate the input combination is unrealistic or outside the model's training data. Setting price to $0."
        )
        prediction = 0.0
    st.success(f"Predicted Price: ${prediction:,.2f}")
