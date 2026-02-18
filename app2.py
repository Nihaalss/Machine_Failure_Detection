import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

# 1. Load the saved model and features
model = joblib.load('lgbm_machine_model.pkl')
features = joblib.load('feature_names.pkl')

# Page Configuration
st.set_page_config(
    page_title="Factory Monitor AI Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ultra-fancy styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .stApp {
        background: linear-gradient(to bottom right, #0f0c29, #302b63, #24243e);
    }
    
    h1 {
        font-family: 'Orbitron', sans-serif;
        color: #00f0ff;
        text-align: center;
        text-shadow: 0 0 20px rgba(0, 240, 255, 0.5);
        font-weight: 900;
        letter-spacing: 3px;
    }
    
    .subtitle {
        text-align: center;
        color: #ffffff;
        font-size: 18px;
        font-family: 'Orbitron', sans-serif;
        margin-bottom: 30px;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    }
    
    .stMetric {
        background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.1));
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(0, 240, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(0, 240, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        font-size: 32px;
        color: #00f0ff;
    }
    
    .status-card {
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        border-left: 5px solid;
        backdrop-filter: blur(10px);
        font-family: 'Orbitron', sans-serif;
    }
    
    .status-critical {
        background: rgba(255, 71, 87, 0.15);
        border-color: #ff4757;
        box-shadow: 0 0 20px rgba(255, 71, 87, 0.3);
    }
    
    .status-warning {
        background: rgba(255, 195, 18, 0.15);
        border-color: #ffc312;
        box-shadow: 0 0 20px rgba(255, 195, 18, 0.3);
    }
    
    .status-normal {
        background: rgba(26, 188, 156, 0.15);
        border-color: #1abc9c;
        box-shadow: 0 0 20px rgba(26, 188, 156, 0.3);
    }
    
    .gauge-container {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(0, 240, 255, 0.3);
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .gauge-title {
        font-family: 'Orbitron', sans-serif;
        color: #00f0ff;
        font-size: 18px;
        margin-bottom: 10px;
    }
    
    .gauge-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 48px;
        font-weight: 900;
        margin: 20px 0;
    }
    
    .gauge-critical { color: #ff4757; }
    .gauge-warning { color: #ffc312; }
    .gauge-normal { color: #1abc9c; }
    
    .progress-bar {
        width: 100%;
        height: 30px;
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        overflow: hidden;
        margin: 10px 0;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 15px;
        transition: all 0.3s ease;
    }
    
    .sensor-card {
        background: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
        margin: 10px 0;
    }
    
    .sensor-label {
        color: #00f0ff;
        font-family: 'Orbitron', sans-serif;
        font-size: 14px;
        margin-bottom: 5px;
    }
    
    .sensor-value {
        color: white;
        font-family: 'Orbitron', sans-serif;
        font-size: 24px;
        font-weight: bold;
    }
    
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'timestamps' not in st.session_state:
    st.session_state.timestamps = []

# Preset configurations
PRESETS = {
    "🟢 Optimal Performance": {
        'footfall': 50, 'tempMode': 45, 'AQ': 80, 'USS': 2.5,
        'CS': 5.0, 'VOC': 150, 'RP': 150, 'IP': 6.0, 'Temperature': 65
    },
    "🟡 Moderate Load": {
        'footfall': 70, 'tempMode': 50, 'AQ': 100, 'USS': 3.5,
        'CS': 7.0, 'VOC': 300, 'RP': 175, 'IP': 7.5, 'Temperature': 80
    },
    "🔴 Critical Stress": {
        'footfall': 95, 'tempMode': 35, 'AQ': 140, 'USS': 4.8,
        'CS': 9.5, 'VOC': 480, 'RP': 195, 'IP': 9.8, 'Temperature': 98
    },
    "❄️ Cold Start": {
        'footfall': 10, 'tempMode': 55, 'AQ': 60, 'USS': 1.2,
        'CS': 2.0, 'VOC': 50, 'RP': 110, 'IP': 3.0, 'Temperature': 52
    }
}

# Title Section
st.markdown("<h1>🛠️ MACHINE FAILURE MONITOR PRO</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>⚡ AI-Powered Predictive Maintenance & Real-Time Diagnostics ⚡</p>", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### 🎮 Control Center")
    st.markdown("---")
    
    # Preset Selection
    st.markdown("#### 🎯 Quick Presets")
    preset_cols = st.columns(2)
    
    preset_selection = None
    for idx, (preset_name, preset_values) in enumerate(PRESETS.items()):
        col = preset_cols[idx % 2]
        if col.button(preset_name, key=f"preset_{idx}", use_container_width=True):
            preset_selection = preset_values
    
    st.markdown("---")
    
    # Initialize values (either from preset or defaults)
    default_values = preset_selection if preset_selection else {
        'footfall': 50, 'tempMode': 45, 'AQ': 100, 'USS': 3.0,
        'CS': 5.0, 'VOC': 250, 'RP': 150, 'IP': 6.0, 'Temperature': 75
    }
    
    # Input Controls with Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Basic", "🔧 Advanced", "⚡ Power"])
    
    with tab1:
        st.markdown("##### Environmental Sensors")
        footfall = st.slider(
            "👥 Footfall",
            0, 100, default_values['footfall'],
            help="Activity level in operational area"
        )
        
        tempMode = st.slider(
            "🌡️ Target Temp (°C)",
            30, 60, default_values['tempMode'],
            help="Desired operating temperature"
        )
        
        AQ = st.slider(
            "💨 Air Quality",
            50, 150, default_values['AQ'],
            help="Ambient air quality index"
        )
    
    with tab2:
        st.markdown("##### Mechanical Sensors")
        USS = st.slider(
            "📡 Ultrasonic",
            1.0, 5.0, default_values['USS'], 0.1,
            help="Vibration/distance measurement"
        )
        
        CS = st.slider(
            "⚙️ Current (A)",
            1.0, 10.0, default_values['CS'], 0.1,
            help="Electrical current draw"
        )
        
        VOC = st.slider(
            "🧪 VOC (ppm)",
            0, 500, default_values['VOC'],
            help="Volatile organic compounds"
        )
    
    with tab3:
        st.markdown("##### Power Metrics")
        RP = st.slider(
            "🔌 Real Power (W)",
            100, 200, default_values['RP'],
            help="Active power consumption"
        )
        
        IP = st.slider(
            "⚡ Input Power",
            2.0, 10.0, default_values['IP'], 0.1,
            help="Total input power"
        )
        
        Temperature = st.slider(
            "🌡️ Actual Temp (°C)",
            50, 100, default_values['Temperature'],
            help="Current operating temperature"
        )
    
    st.markdown("---")
    
    # Action Buttons
    col_btn1, col_btn2 = st.columns(2)
    predict_button = col_btn1.button("🔮 ANALYZE", use_container_width=True, type="primary")
    reset_button = col_btn2.button("🔄 RESET", use_container_width=True)
    
    if reset_button:
        st.session_state.history = []
        st.session_state.predictions = []
        st.session_state.timestamps = []
        st.rerun()

# Function to create custom gauge using HTML/CSS
def create_gauge_html(value, title, max_value=100, thresholds=[40, 70]):
    percentage = (value / max_value) * 100
    
    if value < thresholds[0]:
        color = "#1abc9c"
        status = "NORMAL"
    elif value < thresholds[1]:
        color = "#ffc312"
        status = "WARNING"
    else:
        color = "#ff4757"
        status = "CRITICAL"
    
    return f"""
    <div class="gauge-container">
        <div class="gauge-title">{title}</div>
        <div class="gauge-value" style="color: {color};">{value:.1f}</div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {min(percentage, 100)}%; background: {color};"></div>
        </div>
        <div style="color: {color}; font-family: 'Orbitron'; font-size: 14px; margin-top: 10px;">{status}</div>
    </div>
    """

# Prediction Function
def predict_failure(data):
    data['Power_Efficiency'] = data['RP'] / (data['IP'] + 1e-5)
    data['Thermal_Stress'] = data['Temperature'] - data['tempMode']
    data['Mechanical_Strain'] = data['USS'] * data['CS']
    
    df_input = pd.DataFrame([data])[features]
    prob = model.predict(df_input)[0]
    
    return prob, data

# Main Analysis Section
if predict_button:
    input_data = {
        'footfall': footfall, 'tempMode': tempMode, 'AQ': AQ, 'USS': USS,
        'CS': CS, 'VOC': VOC, 'RP': RP, 'IP': IP, 'Temperature': Temperature
    }
    
    prob, enriched_data = predict_failure(input_data)
    
    # Store history
    st.session_state.predictions.append(prob)
    st.session_state.timestamps.append(datetime.now().strftime("%H:%M:%S"))
    st.session_state.history.append(enriched_data)
    
    # Main Gauge Display
    st.markdown("### 🎯 FAILURE RISK ANALYSIS")
    
    gauge_cols = st.columns(4)
    
    with gauge_cols[0]:
        risk_percentage = prob * 100
        st.markdown(create_gauge_html(risk_percentage, "FAILURE RISK (%)", 100, [40, 70]), unsafe_allow_html=True)
    
    with gauge_cols[1]:
        efficiency = enriched_data['Power_Efficiency']
        st.markdown(create_gauge_html(efficiency, "EFFICIENCY", 50, [15, 30]), unsafe_allow_html=True)
    
    with gauge_cols[2]:
        thermal_stress = abs(enriched_data['Thermal_Stress'])
        st.markdown(create_gauge_html(thermal_stress, "THERMAL (°C)", 50, [15, 30]), unsafe_allow_html=True)
    
    with gauge_cols[3]:
        mechanical_strain = enriched_data['Mechanical_Strain']
        st.markdown(create_gauge_html(mechanical_strain, "STRAIN", 50, [20, 35]), unsafe_allow_html=True)
    
    # Status Alert
    st.markdown("---")
    if prob > 0.7:
        st.markdown(f"""
        <div class="status-card status-critical">
            <h3>🚨 CRITICAL ALERT</h3>
            <p><strong>Risk Level: {risk_percentage:.1f}%</strong></p>
            <p>⚠️ Immediate maintenance required! System failure imminent.</p>
            <p>📋 Recommended Action: Shut down and inspect immediately</p>
        </div>
        """, unsafe_allow_html=True)
    elif prob > 0.4:
        st.markdown(f"""
        <div class="status-card status-warning">
            <h3>⚠️ WARNING</h3>
            <p><strong>Risk Level: {risk_percentage:.1f}%</strong></p>
            <p>🔧 Machine showing elevated stress indicators</p>
            <p>📋 Recommended Action: Schedule preventive maintenance</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-card status-normal">
            <h3>✅ ALL SYSTEMS OPERATIONAL</h3>
            <p><strong>Risk Level: {risk_percentage:.1f}%</strong></p>
            <p>🎯 Machine operating within normal parameters</p>
            <p>📋 Recommended Action: Continue routine monitoring</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed Metrics Grid
    st.markdown("---")
    st.markdown("### 📊 DETAILED DIAGNOSTICS")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("👥 Footfall", footfall, "Activity Level")
        st.metric("🌡️ Target Temp", f"{tempMode}°C", "Set Point")
    
    with metric_col2:
        st.metric("💨 Air Quality", AQ, "Index")
        st.metric("📡 Ultrasonic", f"{USS:.1f}", "Vibration")
    
    with metric_col3:
        st.metric("⚙️ Current", f"{CS:.1f}A", "Draw")
        st.metric("🧪 VOC", f"{VOC} ppm", "Emissions")
    
    with metric_col4:
        st.metric("🔌 Real Power", f"{RP}W", "Consumption")
        st.metric("🌡️ Actual Temp", f"{Temperature}°C", "Current")
    
    # Sensor Readings Display
    st.markdown("---")
    st.markdown("### 🔬 SENSOR READINGS")
    
    sensor_col1, sensor_col2, sensor_col3 = st.columns(3)
    
    with sensor_col1:
        st.markdown(f"""
        <div class="sensor-card">
            <div class="sensor-label">⚡ POWER EFFICIENCY</div>
            <div class="sensor-value">{enriched_data['Power_Efficiency']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with sensor_col2:
        st.markdown(f"""
        <div class="sensor-card">
            <div class="sensor-label">🌡️ THERMAL STRESS</div>
            <div class="sensor-value">{enriched_data['Thermal_Stress']:.2f}°C</div>
        </div>
        """, unsafe_allow_html=True)
    
    with sensor_col3:
        st.markdown(f"""
        <div class="sensor-card">
            <div class="sensor-label">⚙️ MECHANICAL STRAIN</div>
            <div class="sensor-value">{enriched_data['Mechanical_Strain']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

# Historical Analysis
if len(st.session_state.predictions) > 0:
    st.markdown("---")
    st.markdown("### 📈 HISTORICAL TREND ANALYSIS")
    
    # Create DataFrame for chart
    trend_data = pd.DataFrame({
        'Analysis': list(range(1, len(st.session_state.predictions) + 1)),
        'Risk %': [p * 100 for p in st.session_state.predictions]
    })
    
    # Use Streamlit's native line chart
    st.line_chart(
        trend_data.set_index('Analysis'),
        height=400,
        use_container_width=True
    )
    
    # Add threshold reference
    st.markdown("""
    <div style='display: flex; justify-content: center; gap: 30px; margin: 20px 0; font-family: Orbitron;'>
        <span style='color: #1abc9c;'>🟢 Safe Zone: 0-40%</span>
        <span style='color: #ffc312;'>🟡 Warning Zone: 40-70%</span>
        <span style='color: #ff4757;'>🔴 Critical Zone: 70-100%</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistics Dashboard
    st.markdown("### 📊 STATISTICS")
    stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
    
    with stat_col1:
        st.metric("📊 Total Scans", len(st.session_state.predictions))
    
    with stat_col2:
        avg_risk = np.mean(st.session_state.predictions) * 100
        st.metric("📈 Avg Risk", f"{avg_risk:.1f}%")
    
    with stat_col3:
        max_risk = np.max(st.session_state.predictions) * 100
        st.metric("🔴 Peak Risk", f"{max_risk:.1f}%")
    
    with stat_col4:
        min_risk = np.min(st.session_state.predictions) * 100
        st.metric("🟢 Min Risk", f"{min_risk:.1f}%")
    
    with stat_col5:
        critical_count = sum(1 for p in st.session_state.predictions if p > 0.7)
        st.metric("⚠️ Critical", critical_count)
    
    # Historical Data Table
    if st.checkbox("📋 Show Detailed History"):
        history_df = pd.DataFrame({
            'Timestamp': st.session_state.timestamps,
            'Risk %': [f"{p*100:.1f}%" for p in st.session_state.predictions],
            'Status': ['🔴 Critical' if p > 0.7 else '🟡 Warning' if p > 0.4 else '🟢 Normal' 
                      for p in st.session_state.predictions]
        })
        st.dataframe(history_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #00f0ff; padding: 30px; font-family: Orbitron;'>
    <h3>🤖 POWERED BY LIGHTGBM NEURAL NETWORK</h3>
    <p style='color: #888;'>Advanced Machine Learning | Predictive Analytics | Real-Time Monitoring</p>
    <p style='color: #888; font-size: 12px;'>© 2024 Factory Monitor AI Pro | Industrial IoT Division</p>
</div>
""", unsafe_allow_html=True)