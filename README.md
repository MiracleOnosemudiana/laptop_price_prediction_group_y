#  Laptop Price Prediction– Group Project

##  Project Overview
This project is a **Mini-Project in Machine Learning** that aims to predict the selling price of laptops based on their specifications.  
It demonstrates an **end-to-end ML pipeline**, including **data preprocessing, exploratory data analysis (EDA), feature engineering, model building, evaluation, and deployment** using **Streamlit**.

---

##  Objectives
- Apply data cleaning and preprocessing techniques.
- Explore the dataset through EDA and visualizations.
- Engineer meaningful features to improve predictions.
- Build and evaluate a **Linear Regression model**.
- Deploy the model as a **Streamlit web app** for user-friendly predictions.
- Maintain the project with proper **version control (Git & GitHub)**.

---

##  Dataset
We use the **Laptop Price Dataset** from Kaggle:  
 [Laptop Price Dataset](https://www.kaggle.com/datasets/muhammetvarl/laptop-price)

**Features include:**
- Company, Product, TypeName  
- Screen size (inches), Resolution  
- CPU, RAM, Memory  
- GPU, Operating System  
- Weight, Price (target variable)

---

##  Project Structure

laptop_price_prediction/
│── model/
│ ├── laptop_price_model.pkl
│ ├── model_features.pkl
│ ├── train_test_split.pkl
│
│── data/
│ |── final/  # Final feature-engineered dataset
│ ├── encoded/ # Encoded datasets
│ ├── processed/ # Cleaned datasets
│ ├── raw/ # Original dataset
│
│── notebooks/ # Jupyter notebooks for each stage
│ ├── 01_data_cleaning.ipynb
│ ├── 02_eda.ipynb
│ ├── 03_feature_engineering.ipynb
│ ├── 04_model_building.ipynb
│ ├── 05_model_evaluation.ipynb
│
│── src/ # Python scripts (modular code)
│ ├── data_preprocessing.py
│ ├── eda.py
│ ├── features.py
│ ├── model.py
│ ├── evaluation.py
│
│── streamlit_app/ # Streamlit deployment
│ ├── app.py
│ ├── requirements.txt
│
│── reports/ # Documentation
│ ├── final_report.docx/pdf
│ ├── eda_visualizations/
│
│── README.md # Project overview
│── .gitignore # Ignore unnecessary files



---

##  Team Roles (9 Members)
- **Data Cleaning (2)** → Handle missing values, outliers, convert RAM/Weight/Memory.  
- **EDA (2)** → Distributions, visualizations, correlation heatmaps.  
- **Feature Engineering (1)** → Extract CPU/GPU brand, storage type, PPI.  
- **Model Building (2)** → Train-test split, build Linear Regression model.  
- **Model Evaluation (1)** → Evaluate using RMSE, MAE, R².  
- **Deployment (1)** → Develop Streamlit app & deployment.  
- **Documentation (shared)** → Compile final report and presentation.  

---

##  Kanban Workflow (GitHub Project Board)
###  Columns
- **To Do**  – tasks not started.  
- **In Progress**  – tasks someone is working on.  
- **Review** – waiting for review/merge.  
- **Done**  – completed tasks.  
- **Blocked** – if someone is stuck.  

###  Task Breakdown
- Note that:-
    Random State = 9
    Test Size (%) = 25

#### Data Cleaning
- Handle missing values  
- Handle outliers  
- Convert RAM, Memory, and Weight to numeric  

#### EDA
- Plot distributions (histograms/boxplots)  
- Correlation heatmap  
- Identify top factors affecting price  

#### Feature Engineering
- Extract CPU brand & clock speed  
- Extract GPU brand  
- Convert Resolution → Pixels Per Inch (PPI)  
- Encode categorical features  

#### Model Building
- Train-test split  
- Build Linear Regression baseline model  
- Save trained model  

#### Model Evaluation
- Calculate RMSE, MAE, R²  
- Compare training vs test performance  

#### Deployment
- Build Streamlit app UI  
- Add input widgets (dropdowns, sliders)  
- Connect model to app for predictions  
- Deploy to Streamlit Cloud  

#### Documentation
- EDA summary  
- Model performance report  
- Final 10-page report  

---

## Setup Instructions for Collaborators

**Step 0 – Prerequisites**  
- Install **Python 3.10+** on your system.  
- Make sure you have **Git** installed.  

**Step 1 – Clone the repository**  
- Run: git clone https://github.com/MiracleOnosemudiana/laptop_price_prediction_group_y 
- Navigate into the folder: cd laptop_price_prediction_group_y  

**Step 2 – Create virtual environment**  
- Run: python -m venv venv  
- Activate it:  
  - On Windows (PowerShell): .\venv\Scripts\activate  
  - On macOS/Linux: source venv/bin/activate  

**Step 3 – Install dependencies**  
- Run: pip install -r requirements.txt  

**Step 4 – Work on Jupyter Notebooks**  
- Launch: jupyter notebook or visual studio code  
- Open files inside the **notebooks/** folder.  

**Step 5 – Run the Streamlit app**  
- Run: streamlit run app.py  
- The app will open in your browser.  

---

##  Collaboration Workflow

**Branching Strategy**  
- Always create a new branch before starting work. Example:  
  git checkout -b feature-data-cleaning  

**Commit Messages**  
- Use clear, short commit messages. Example:  
  Added preprocessing for missing values  

**Push Changes**  
- Push your branch: git push origin feature-data-cleaning  

**Pull Requests**  
- Open a Pull Request (PR) for review before merging.  

---

##  How to Run
### 1. Clone the repo
```bash
git clone https://github.com/your_username/laptop_price_prediction.git
cd laptop_price_prediction