# 🛠️ Machine Failure Prediction & Real-Time Monitoring System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-ML-green.svg)](https://lightgbm.readthedocs.io/)

An AI-powered predictive maintenance system that monitors industrial machinery in real-time and predicts failure probability using machine learning.

![Dashboard Preview](https://nihaalss-machine-failure-detection-app2-0uicql.streamlit.app/)



---

## 🎯 Overview

This project combines **machine learning** and **interactive data visualization** to create a predictive maintenance solution for industrial equipment. By analyzing real-time sensor data, the system can:

- Predict machine failure probability with **high accuracy**
- Provide actionable maintenance recommendations
- Track equipment health over time
- Alert operators to critical conditions before failures occur

**Key Achievement:** Successfully reduced potential downtime by enabling proactive maintenance scheduling based on ML predictions.

---

## ✨ Features

### 🤖 Machine Learning Pipeline
- **Exploratory Data Analysis (EDA):** Comprehensive statistical analysis and visualization
- **Feature Engineering:** Created advanced features (Power Efficiency, Thermal Stress, Mechanical Strain)
- **Model Training:** LightGBM gradient boosting classifier
- **Model Evaluation:** Model Performance

### 🌐 Interactive Dashboard
- **Real-Time Monitoring:** Live sensor input via interactive sliders
- **Visual Gauges:** Color-coded risk indicators (Green/Yellow/Red)
- **Quick Presets:** Pre-configured scenarios for testing
- **Historical Tracking:** Trend analysis with line charts
- **Smart Alerts:** Context-aware notifications with actionable recommendations
- **Responsive Design:** Cyberpunk-inspired UI with glassmorphism effects

### 📊 Monitored Sensors (9 inputs)
| Sensor | Description | Range | Impact |
|--------|-------------|-------|--------|
| 👥 Footfall | Activity/usage level | 0-100 | High usage → More wear |
| 🌡️ Temperature Mode | Target temperature | 30-60°C | Affects thermal stress |
| 💨 Air Quality | Environmental cleanliness | 50-150 AQI | Poor quality → Dust/clog |
| 📡 Ultrasonic | Vibration/movement | 1.0-5.0 | High vibration → Failure |
| ⚙️ Current Sensor | Electrical load | 1.0-10.0 A | High current → Stress |
| 🧪 VOC | Chemical exposure | 0-500 ppm | Degrades components |
| 🔌 Real Power | Active power | 100-200 W | Efficiency indicator |
| ⚡ Input Power | Total power supplied | 2.0-10.0 | Power draw |
| 🌡️ Temperature | Actual temperature | 50-100°C | Overheating risk |

---

## 🏗️ Model Architecture

### Feature Engineering
Created three advanced features combining sensor readings:

1. **Power Efficiency = Real Power / Input Power**
   - Measures energy conversion efficiency
   - Values below 15 indicate potential issues

2. **Thermal Stress = Actual Temperature - Target Temperature**
   - Quantifies temperature deviation
   - High absolute values indicate cooling/heating failure

3. **Mechanical Strain = Ultrasonic × Current Sensor**
   - Captures combined mechanical and electrical stress
   - Values above 35 indicate critical condition

### Model Selection & Training

**Algorithm:** LightGBM (Light Gradient Boosting Machine)

**Why LightGBM?**
- Handles mixed data types efficiently
- Fast training on large datasets
- Built-in handling of missing values
- Superior performance on imbalanced data
- Feature importance extraction

**Confusion Matrix:**
```
                    Predicted
                  Fail | No Fail
Actual  Fail      140  |  19
        No Fail    27  |  3
```

##💻 Running the Dashboard
### Using the Interface

#### **Option 1: Quick Presets**
1. Open the sidebar (left panel)
2. Click any preset button:
   - 🟢 **Optimal Performance** - Normal operating conditions
   - 🟡 **Moderate Load** - Elevated but safe conditions
   - 🔴 **Critical Stress** - High-risk scenario
   - ❄️ **Cold Start** - Low-load startup conditions
3. Click **🔮 ANALYZE** button

#### **Option 2: Manual Input**
1. Open sidebar and navigate through tabs:
   - **📊 Basic:** Footfall, Target Temp, Air Quality
   - **🔧 Advanced:** Ultrasonic, Current, VOC
   - **⚡ Power:** Real Power, Input Power, Actual Temp
2. Adjust sliders to desired values
3. Click **🔮 ANALYZE** button

#### **Understanding Results**

**Risk Levels:**
- 🟢 **0-40%:** Safe - Continue routine monitoring
- 🟡 **40-70%:** Warning - Schedule preventive maintenance
- 🔴 **70-100%:** Critical - Immediate action required!

**Dashboard Sections:**
1. **Gauges:** Visual risk indicators for failure, efficiency, thermal, and strain
2. **Status Card:** Color-coded alert with recommendations
3. **Diagnostics:** Detailed sensor readings
4. **Sensor Readings:** Calculated engineered features
5. **Historical Trends:** Line chart showing risk over multiple analyses
6. **Statistics:** Summary metrics (avg, peak, min risk)

---

## 📁 Project Structure

```
machine-failure/
│
├── machine_monitor_final.py          # Main Streamlit dashboard
├── lgbm_machine_model.pkl            # Trained LightGBM model
├── feature_names.pkl                 # Feature order for model
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── COMPLETE_GUIDE.md                 # Detailed documentation
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_data_cleaning.ipynb       # Data preprocessing
│   ├── 02_eda.ipynb                 # Exploratory analysis
│   ├── 03_feature_engineering.ipynb # Feature creation
│   └── 04_model_training.ipynb      # Model development
│
├── data/                            # Data directory
│   ├── raw/                         # Original datasets
│   ├── processed/                   # Cleaned datasets
│   └── README.md                    # Data description
│
├── models/                          # Saved models
│   ├── lgbm_machine_model.pkl      # Production model
│   └── model_metrics.json          # Performance metrics
│
├── images/                          # Screenshots & visuals
│   ├── dashboard.png
│   ├── eda_plots.png
│   └── confusion_matrix.png
│
└── docs/                            # Additional documentation
    ├── COMPLETE_GUIDE.md
    └── API_REFERENCE.md
```

---

## 📸 Screenshots

### Dashboard Overview
![Dashboard](https://via.placeholder.com/800x450?text=Main+Dashboard+View)

### Gauge Indicators
![Gauges](https://via.placeholder.com/800x300?text=Risk+Gauges+Display)

### Historical Trends
![Trends](https://via.placeholder.com/800x400?text=Historical+Trend+Chart)

### EDA Visualizations
![EDA](https://via.placeholder.com/800x400?text=Correlation+Heatmap+and+Distributions)

---

## 🛠️ Technologies Used

### Machine Learning & Data Science
- **Python 3.8+** - Core programming language
- **LightGBM** - Gradient boosting framework
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - ML utilities and metrics
- **Matplotlib/Seaborn** - Data visualization (EDA)

### Web Application
- **Streamlit** - Interactive dashboard framework
- **Joblib** - Model serialization

### Development Tools
- **Jupyter Notebook** - Interactive development
- **Git** - Version control
- **VS Code** - Code editor

---

## ⭐ Show Your Support

If this project helped you, please give it a ⭐ on GitHub!

---

*Last Updated: February 2024*
