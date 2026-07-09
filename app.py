import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import textwrap

def render_html_card(html_str):
    # Strip leading/trailing spaces from each line and join without newlines
    clean_str = "".join([line.strip() for line in html_str.split("\n")])
    st.markdown(clean_str, unsafe_allow_html=True)


# Set page config
st.set_page_config(
    page_title="Health Prediction Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set matplotlib style for dark mode compatibility or clean look
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['text.color'] = '#1e293b'
plt.rcParams['axes.labelcolor'] = '#1e293b'
plt.rcParams['xtick.color'] = '#1e293b'
plt.rcParams['ytick.color'] = '#1e293b'

# ==========================================
# CACHED RESOURCE LOADING
# ==========================================
@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_data
def load_metrics(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

# ==========================================
# NAVIGATION STATE
# ==========================================
if "page" not in st.session_state:
    st.session_state.page = "Symptom-Based"

# ==========================================
# THEME STYLING FUNCTION
# ==========================================
def apply_theme_css(theme_color, bg_start, bg_end):
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp {{
        font-family: 'Outfit', sans-serif;
    }}
    
    /* Background Gradient */
    .stApp {{
        background: linear-gradient(135deg, {bg_start} 0%, {bg_end} 100%) !important;
    }}
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span {{
        color: #f8fafc !important;
    }}
    
    /* Style for sidebar buttons */
    section[data-testid="stSidebar"] button[kind="secondary"] {{
        background-color: rgba(255, 255, 255, 0.04) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        text-align: left !important;
        padding: 8px 16px !important;
        transition: all 0.25s !important;
    }}
    
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {{
        background-color: {theme_color}25 !important;
        border-color: {theme_color} !important;
        color: #ffffff !important;
        transform: translateX(4px) !important;
    }}
    
    section[data-testid="stSidebar"] button[kind="primary"] {{
        background-color: {theme_color} !important;
        color: #0f172a !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 8px 16px !important;
        box-shadow: 0 4px 12px {theme_color}44 !important;
    }}
    
    /* Primary buttons in main page */
    [data-testid="stMain"] div.stButton button[kind="primary"] {{
        background-color: {theme_color} !important;
        color: #0f172a !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 28px !important;
        box-shadow: 0 4px 15px {theme_color}44 !important;
        transition: all 0.3s !important;
    }}
    
    [data-testid="stMain"] div.stButton button[kind="primary"]:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 22px {theme_color}66 !important;
        color: #0f172a !important;
    }}
    
    /* Glassmorphism Cards */
    .glass-card {{
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(14px);
        border-radius: 16px;
        padding: 26px;
        border: 1px solid rgba(255, 255, 255, 0.45);
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.03);
        margin-bottom: 24px;
        color: #1e293b;
    }}
    
    .home-card {{
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(14px);
        border-radius: 16px;
        padding: 26px;
        border: 1px solid rgba(255, 255, 255, 0.45);
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.03);
        transition: all 0.3s ease;
        min-height: 240px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    
    .home-card:hover {{
        transform: translateY(-6px);
        box-shadow: 0 15px 35px rgba(15, 23, 42, 0.08);
    }}
    
    /* Typography customization (Solid Black headings) */
    h1, h2, h3, [data-testid="stHeader"] h1, [data-testid="stHeader"] h2, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6, div[data-testid="stExpander"] p, div.glass-card h3, div.home-card h3 {{
        color: #000000 !important;
        font-weight: 700 !important;
    }}
    
    /* Non-sidebar text visibility override - Forces light theme text/labels */
    [data-testid="stMain"] p,
    [data-testid="stMain"] span,
    [data-testid="stMain"] label,
    [data-testid="stMain"] div.stMarkdown,
    [data-testid="stMain"] table,
    [data-testid="stMain"] td,
    [data-testid="stMain"] th,
    [data-testid="stMain"] tr,
    [data-testid="stMain"] li,
    div[data-testid="stMetricValue"] div,
    div[data-testid="stMetricLabel"] div,
    .stApp [data-testid="stHeader"] {{
        color: #1e293b !important;
    }}
    
    /* Table Legibility overrides */
    [data-testid="stMain"] table {{
        background-color: #ffffff !important;
    }}
    [data-testid="stMain"] th {{
        font-weight: 700 !important;
        background-color: rgba(140, 240, 214, 0.1) !important;
        color: #000000 !important;
    }}
    [data-testid="stMain"] td {{
        color: #1e293b !important;
    }}
    
    .badge {{
        background-color: {theme_color}20;
        border: 1px solid {theme_color};
        color: #1e293b;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        display: inline-block;
        margin-bottom: 8px;
    }}
    
    /* Input customization - Set solid white background and clear black text */
    .stTextInput input, .stNumberInput input, .stTextInput > div, .stNumberInput > div, div[data-baseweb="select"] {{
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid rgba(0, 0, 0, 0.22) !important;
        border-radius: 8px !important;
    }}
    
    .stTextInput input, .stNumberInput input {{
        color: #000000 !important;
    }}
    
    /* Dropdown select box labels and values */
    div[data-baseweb="select"] * {{
        background-color: #ffffff !important;
        color: #000000 !important;
    }}
    
    /* Dropdown popover options list when expanded */
    div[data-baseweb="popover"] *, ul[role="listbox"] *, li[role="option"] * {{
        background-color: #ffffff !important;
        color: #000000 !important;
    }}
    li[role="option"]:hover, li[role="option"]:hover * {{
        background-color: #8CF0D6 !important;
        color: #0a5843 !important;
    }}
    
    /* Sidebar text visibility overrides - preserves dark sidebar text */
    section[data-testid="stSidebar"] *, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] button,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] a {{
        color: #f8fafc !important;
    }}
    
    .stSlider div[data-testid="stSliderTickBar"] {{
        color: #475569 !important;
    }}
    
    /* Custom divider */
    .custom-hr {{
        border: 0;
        height: 1px;
        background: linear-gradient(to right, rgba(15, 23, 42, 0), rgba(15, 23, 42, 0.15), rgba(15, 23, 42, 0));
        margin: 25px 0;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def apply_symptom_page_css():
    css = """
    <style>
    /* Styling for primary buttons (selected chips and grid items) in main area */
    [data-testid="stMain"] div.stButton button[kind="primary"] {
        background-color: #8CF0D6 !important;
        color: #0a5843 !important;
        border: 1px solid #75dfc4 !important;
        border-radius: 24px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 5px 12px !important;
        box-shadow: 0 2px 5px rgba(140, 240, 214, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stMain"] div.stButton button[kind="primary"]:hover {
        background-color: #ef4444 !important;
        border-color: #ef4444 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 10px rgba(239, 68, 68, 0.3) !important;
    }
    
    /* Styling for secondary buttons (unselected grid items) - solid white, dark border and text */
    [data-testid="stMain"] div.stButton button[kind="secondary"] {
        background-color: #ffffff !important;
        color: #1e293b !important;
        border: 1px solid rgba(0, 0, 0, 0.22) !important;
        border-radius: 24px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 5px 12px !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stMain"] div.stButton button[kind="secondary"]:hover {
        background-color: #8CF0D633 !important;
        border-color: #8CF0D6 !important;
        color: #0a5843 !important;
    }
    
    /* Prevent the Analyze button from turning red on hover */
    /* Target the primary button that is the last child element of the first column */
    [data-testid="stMain"] div[data-testid="column"]:first-child > div:last-child div.stButton button[kind="primary"]:hover {
        background-color: #6ce0c2 !important;
        border-color: #56cca9 !important;
        color: #0a5843 !important;
        box-shadow: 0 4px 12px rgba(140, 240, 214, 0.4) !important;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ==========================================
# VISUALIZATION HELPERS
# ==========================================
def plot_confusion_matrix(cm, labels, theme_color):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    cmap = sns.light_palette(theme_color, as_cmap=True)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
    ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=9)
    ax.set_ylabel('True Label', fontweight='bold', fontsize=9)
    ax.set_title('Confusion Matrix', fontweight='bold', fontsize=10, pad=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    return fig

def plot_roc_curves(metrics_dict, selected_model_name, theme_color):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Guess (0.50)')
    
    for name, metrics in metrics_dict.items():
        if "roc_curve" in metrics:
            fpr = metrics["roc_curve"]["fpr"]
            tpr = metrics["roc_curve"]["tpr"]
            auc = metrics["roc_auc"]
            
            # Highlight selected
            if name == selected_model_name:
                ax.plot(fpr, tpr, color=theme_color, lw=2.5, label=f"{name} (AUC = {auc:.3f})")
            else:
                ax.plot(fpr, tpr, alpha=0.4, lw=1.2, label=f"{name} (AUC = {auc:.3f})")
                
    ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=9)
    ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=9)
    ax.set_title('ROC Curve Comparison', fontweight='bold', fontsize=10, pad=10)
    ax.legend(loc='lower right', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    return fig

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; margin-bottom: 0;'>🏥 HealthPredictor</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 14px;'>Clinical ML Dashboard</p>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    
    nav_items = {
        "Symptom-Based": "🩺 Symptom Checker",
        "Heart Disease": "❤️ Heart Disease Risk",
        "Mental Health": "🧠 Mental Health Risk",
        "Diabetes": "🩸 Diabetes Prediction"
    }
    
    for key, label in nav_items.items():
        is_active = st.session_state.page == key
        btn_type = "primary" if is_active else "secondary"
        if st.button(label, key=f"nav_{key}", use_container_width=True, type=btn_type):
            st.session_state.page = key
            st.rerun()
            
    st.markdown("<div style='position: fixed; bottom: 15px; width: 220px; text-align: center; color: #64748b; font-size: 11px;'>HealthPredictor ML Framework &copy; 2026</div>", unsafe_allow_html=True)


# ==========================================
# PAGE ROUTING
# ==========================================

# ------------------------------------------
# PAGE: SYMPTOM-BASED DISEASE PREDICTION
# ------------------------------------------
if st.session_state.page == "Symptom-Based":
    apply_theme_css("#0cbd8f", "#f4fbf7", "#e6f7f0")
    apply_symptom_page_css()
    
    render_html_card("""
        <div style="
            background: linear-gradient(135deg, #0cbd8f 0%, #059669 100%);
            border-radius: 16px;
            padding: 24px;
            color: #ffffff;
            box-shadow: 0 10px 25px rgba(12, 189, 143, 0.15);
            margin-bottom: 30px;
            display: flex;
            align-items: center;
            gap: 20px;
            position: relative;
            overflow: hidden;
            margin-top: 10px;
        ">
            <!-- Decorative background shapes -->
            <div style="position: absolute; right: -30px; top: -30px; width: 120px; height: 120px; border-radius: 50%; background: rgba(255, 255, 255, 0.1); pointer-events: none;"></div>
            <div style="position: absolute; right: 50px; bottom: -50px; width: 100px; height: 100px; border-radius: 50%; background: rgba(255, 255, 255, 0.05); pointer-events: none;"></div>
            
            <!-- Icon Container -->
            <div style="
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 12px;
                padding: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
            ">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="color: white;">
                    <path d="M12 2v10a4 4 0 0 0 8 0V4"/>
                    <path d="M12 12a4 4 0 0 1-8 0V4"/>
                    <path d="M12 4h8"/>
                    <path d="M12 4H4"/>
                    <path d="M12 16v4"/>
                    <circle cx="12" cy="20" r="2"/>
                </svg>
            </div>
            
            <!-- Text Content -->
            <div style="flex-grow: 1;">
                <h1 style="
                    color: #ffffff !important;
                    margin: 0 0 4px 0 !important;
                    font-size: 24px !important;
                    font-weight: 700 !important;
                    line-height: 1.2 !important;
                    border: none !important;
                    padding: 0 !important;
                ">Symptom-Based Disease Prediction</h1>
                <p style="
                    color: rgba(255, 255, 255, 0.9) !important;
                    margin: 0 !important;
                    font-size: 14px !important;
                    font-weight: 400 !important;
                    line-height: 1.4 !important;
                ">Check diagnosis predictions by selecting active patient symptoms.</p>
            </div>
        </div>
    """)
    
    # Load assets
    symptom_features = load_model("models/symptom_features.joblib")
    metrics_data = load_metrics("models/symptom_metrics.json")
    
    # Format symptom labels nicely
    formatted_symptoms = {sym: sym.replace('_', ' ').title() for sym in symptom_features}
    reverse_symptoms = {v: k for k, v in formatted_symptoms.items()}
    
    # Initialize session state variables for Symptom-Based page
    if "selected_symptoms" not in st.session_state:
        st.session_state.selected_symptoms = []
    if "symptom_prediction" not in st.session_state:
        st.session_state.symptom_prediction = None
    if "symptom_model_selected" not in st.session_state:
        st.session_state.symptom_model_selected = list(metrics_data.keys())[0]
        
    def toggle_symptom(sym):
        if sym in st.session_state.selected_symptoms:
            st.session_state.selected_symptoms.remove(sym)
        else:
            st.session_state.selected_symptoms.append(sym)
        # Clear previous prediction since inputs changed
        st.session_state.symptom_prediction = None
    
    # Model Selection at the top of the form
    model_name = st.selectbox(
        "Select Classifier Algorithm:", 
        list(metrics_data.keys()), 
        index=list(metrics_data.keys()).index(st.session_state.symptom_model_selected)
    )
    if model_name != st.session_state.symptom_model_selected:
        st.session_state.symptom_model_selected = model_name
        st.session_state.symptom_prediction = None
        
    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    
    # 2-column layout: Left for selection, Right for prediction output
    col_left, col_right = st.columns([1.2, 0.8])
    
    with col_left:
        st.markdown("""
            <div class="glass-card" style="border-top: 5px solid #8CF0D6; padding: 20px; margin-bottom: 15px;">
                <h3 style="margin-top: 0; margin-bottom: 5px;">Symptom-based diagnosis</h3>
                <p style="color: #475569; font-size: 14px; margin-bottom: 0;">Select every symptom you're experiencing. The system will rank the most likely conditions.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Search Bar
        search_query = st.text_input(
            "Search symptoms:",
            placeholder="🔍 Search symptoms — e.g. headache, cough",
            label_visibility="collapsed"
        )
        
        # Display Selected Symptoms as Removable Tags/Chips
        if st.session_state.selected_symptoms:
            st.markdown("<div style='margin-top: 15px; margin-bottom: 5px; font-size: 13px; font-weight: 600; color: #475569;'>Selected Symptoms (click to remove):</div>", unsafe_allow_html=True)
            
            tags = st.session_state.selected_symptoms
            for i in range(0, len(tags), 4):
                row_tags = tags[i:i+4]
                cols = st.columns(4)
                for j, sym in enumerate(row_tags):
                    disp_name = formatted_symptoms[sym]
                    with cols[j]:
                        if st.button(f"{disp_name} ✕", key=f"chip_{sym}", type="primary", use_container_width=True):
                            toggle_symptom(sym)
                            st.rerun()
        
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        
        # Scrollable Grid of Symptoms
        st.markdown("<div style='font-size: 14px; font-weight: 600; color: #475569; margin-bottom: 8px;'>Available Symptoms:</div>", unsafe_allow_html=True)
        
        # Filter based on search query
        filtered_syms = []
        for sym in symptom_features:
            disp_name = formatted_symptoms[sym]
            if not search_query or search_query.lower() in disp_name.lower():
                filtered_syms.append(sym)
        
        # Scrollable Container
        with st.container(height=300):
            # 3 columns grid
            grid_cols = st.columns(3)
            for idx, sym in enumerate(filtered_syms):
                col_idx = idx % 3
                disp_name = formatted_symptoms[sym]
                is_selected = sym in st.session_state.selected_symptoms
                
                with grid_cols[col_idx]:
                    # Selected gets primary (green), unselected gets secondary (white/gray)
                    if st.button(
                        disp_name,
                        key=f"grid_btn_{sym}",
                        use_container_width=True,
                        type="primary" if is_selected else "secondary"
                    ):
                        toggle_symptom(sym)
                        st.rerun()
        
        # Analyze Button
        num_selected = len(st.session_state.selected_symptoms)
        btn_label = f"Analyze {num_selected} symptom{'s' if num_selected != 1 else ''}"
        
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        if st.button(
            btn_label,
            key="btn_analyze_symptoms",
            use_container_width=True,
            type="primary",
            disabled=(num_selected == 0)
        ):
            # Run prediction
            model_key = st.session_state.symptom_model_selected.lower().replace(" ", "_")
            model = load_model(f"models/symptom_{model_key}.joblib")
            
            if model:
                # Construct feature vector
                input_vector = np.zeros((1, len(symptom_features)))
                for sym in st.session_state.selected_symptoms:
                    idx = symptom_features.index(sym)
                    input_vector[0, idx] = 1
                
                # Predict
                pred = model.predict(input_vector)[0]
                st.session_state.symptom_prediction = pred
                st.rerun()
            else:
                st.error("Error loading model.")
    
    with col_right:
        # If we have a prediction, display it
        if st.session_state.symptom_prediction:
            pred = st.session_state.symptom_prediction
            st.markdown(f"""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; min-height: 400px; text-align: center;">
                    <div class="glass-card" style="border-left: 6px solid #0cbd8f; background-color: #ffffff; text-align: left; width: 100%;">
                        <h3 style="color: #099268; margin-top: 0;">Diagnostic Prediction Results</h3>
                        <p style="font-size: 16px;">The diagnostic model classifies this symptom pattern as:</p>
                        <div style="font-size: 26px; font-weight: 700; color: #0cbd8f; background: #e6fcf5; display: inline-block; padding: 10px 20px; border-radius: 8px; border: 1px solid #0cbd8f; margin-bottom: 15px;">
                            {pred}
                        </div>
                        <p style="font-size: 14px; color: #475569; line-height: 1.6;">
                            Based on the selected symptoms, the system predicts a diagnosis of <strong>{pred}</strong> using the <strong>{st.session_state.symptom_model_selected}</strong> classifier.
                        </p>
                        <p style="font-size: 13px; color: #64748b; margin-top: 20px; border-top: 1px solid rgba(0,0,0,0.05); padding-top: 15px;">
                            *Disclaimer: This analysis is generated by a machine learning model for informational purposes only. Please consult a qualified physician for clinical diagnosis and treatment.*
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Awaiting input placeholder
            st.markdown("""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; min-height: 440px; text-align: center; background: rgba(255, 255, 255, 0.45); border-radius: 16px; border: 1px dashed rgba(15, 23, 42, 0.1); padding: 30px;">
                    <div style="width: 80px; height: 80px; border-radius: 50%; background-color: #8CF0D6; display: flex; align-items: center; justify-content: center; font-size: 36px; font-weight: bold; color: #0a5843; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(140, 240, 214, 0.35);">?</div>
                    <h3 style="color: #000000 !important; font-weight: 700; margin-bottom: 10px; margin-top: 0;">Awaiting your inputs</h3>
                    <p style="color: #475569; max-width: 280px; line-height: 1.5; font-size: 14px; margin: 0;">Add at least one symptom, then let the system rank the top conditions.</p>
                </div>
            """, unsafe_allow_html=True)
                
    st.markdown("<div class='custom-hr'></div>", unsafe_allow_html=True)
    
    if metrics_data:
        # Model metrics table
        st.subheader("📊 Model Performance & Evaluation Metrics")
        rows = []
        for name, data in metrics_data.items():
            rows.append({
                "Model Name": name,
                "Accuracy": f"{data['accuracy']*100:.2f}%",
                "Precision": f"{data['precision']*100:.2f}%",
                "Recall": f"{data['recall']*100:.2f}%",
                "F1 Score": f"{data['f1_score']*100:.2f}%"
            })
        st.table(pd.DataFrame(rows).set_index("Model Name"))
        
        # Detailed stats
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader(f"Detailed Model Confusion Analysis ({model_name})")
        
        selected_metrics = metrics_data[model_name]
        cm = np.array(selected_metrics["confusion_matrix"])
        classes = selected_metrics["classes"]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        cmap = sns.light_palette("#0cbd8f", as_cmap=True)
        sns.heatmap(cm, annot=False, cmap=cmap, xticklabels=classes, yticklabels=classes, ax=ax)
        plt.xticks(rotation=90, fontsize=6)
        plt.yticks(fontsize=6)
        ax.set_xlabel('Predicted Class', fontweight='bold', fontsize=8)
        ax.set_ylabel('True Class', fontweight='bold', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)


# ------------------------------------------
# PAGE: HEART DISEASE PREDICTION
# ------------------------------------------
elif st.session_state.page == "Heart Disease":
    apply_theme_css("#ef4444", "#fdf2f2", "#fff0f0")
    
    render_html_card("""
        <div style="
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            border-radius: 16px;
            padding: 24px;
            color: #ffffff;
            box-shadow: 0 10px 25px rgba(239, 68, 68, 0.15);
            margin-bottom: 30px;
            display: flex;
            align-items: center;
            gap: 20px;
            position: relative;
            overflow: hidden;
            margin-top: 10px;
        ">
            <!-- Decorative background shapes -->
            <div style="position: absolute; right: -30px; top: -30px; width: 120px; height: 120px; border-radius: 50%; background: rgba(255, 255, 255, 0.1); pointer-events: none;"></div>
            <div style="position: absolute; right: 50px; bottom: -50px; width: 100px; height: 100px; border-radius: 50%; background: rgba(255, 255, 255, 0.05); pointer-events: none;"></div>
            
            <!-- Icon Container -->
            <div style="
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 12px;
                padding: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
            ">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="color: white;">
                    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
                    <path d="M6 12h3l2-3 2 6 1-3h4"/>
                </svg>
            </div>
            
            <!-- Text Content -->
            <div style="flex-grow: 1;">
                <h1 style="
                    color: #ffffff !important;
                    margin: 0 0 4px 0 !important;
                    font-size: 24px !important;
                    font-weight: 700 !important;
                    line-height: 1.2 !important;
                    border: none !important;
                    padding: 0 !important;
                ">Heart Disease Prediction</h1>
                <p style="
                    color: rgba(255, 255, 255, 0.9) !important;
                    margin: 0 !important;
                    font-size: 14px !important;
                    font-weight: 400 !important;
                    line-height: 1.4 !important;
                ">Determine cardiac risk levels using clinical measurements.</p>
            </div>
        </div>
    """)
    
    # Load models & configurations
    heart_columns = load_model("models/heart_columns.joblib")
    scaler_heart = load_model("models/heart_scaler.joblib")
    metrics_data = load_metrics("models/heart_metrics.json")
    
    # Model selection placed prominently below the banner
    model_name = st.selectbox("Select ML Model (applies to both prediction and performance stats below):", list(metrics_data.keys()), key="heart_model_select")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Clinical Parameters")
    
    # Form inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Patient Age (Years):", 1, 100, 50)
        sex = st.selectbox("Patient Gender (Sex):", ["Female", "Male"])
        chest_pain = st.selectbox(
            "Chest Pain Type (ChestPainType):",
            ["Asymptomatic (ASY)", "Atypical Angina (ATA)", "Non-Anginal Pain (NAP)", "Typical Angina (TA)"]
        )
        resting_bp = st.number_input("Resting Blood Pressure (RestingBP in mmHg):", min_value=0, max_value=240, value=120)
        
    with col2:
        cholesterol = st.number_input("Serum Cholesterol (Cholesterol in mg/dl):", min_value=0, max_value=600, value=200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (FastingBS):", ["No", "Yes"])
        resting_ecg = st.selectbox(
            "Resting Electrocardiogram (RestingECG):",
            ["Normal", "ST-T wave abnormality (ST)", "Left ventricular hypertrophy (LVH)"]
        )
        
    with col3:
        max_hr = st.slider("Maximum Heart Rate Achieved (MaxHR in bpm):", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise Induced Angina (ExerciseAngina):", ["No", "Yes"])
        oldpeak = st.slider("ST Depression Induced by Exercise (Oldpeak):", 0.0, 6.0, 1.0, step=0.1)
        st_slope = st.selectbox(
            "Slope of Peak Exercise ST Segment (ST_Slope):",
            ["Upsloping (Up)", "Flat", "Downsloping (Down)"]
        )
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Evaluate Cardiac Risk", type="primary"):
        # Setup features dict matching dummy coding
        features_dict = {col: 0 for col in heart_columns}
        
        # Set numerical values (handle 0 BP/Chol replacement values)
        features_dict['Age'] = age
        features_dict['RestingBP'] = resting_bp if resting_bp != 0 else 132.40
        features_dict['Cholesterol'] = cholesterol if cholesterol != 0 else 244.63
        features_dict['FastingBS'] = 1 if fasting_bs == "Yes" else 0
        features_dict['MaxHR'] = max_hr
        features_dict['Oldpeak'] = oldpeak
        
        # Map Categorical
        if sex == "Male":
            features_dict['Sex_M'] = 1
            
        if chest_pain == "Atypical Angina (ATA)":
            features_dict['ChestPainType_ATA'] = 1
        elif chest_pain == "Non-Anginal Pain (NAP)":
            features_dict['ChestPainType_NAP'] = 1
        elif chest_pain == "Typical Angina (TA)":
            features_dict['ChestPainType_TA'] = 1
            
        if resting_ecg == "Normal":
            features_dict['RestingECG_Normal'] = 1
        elif resting_ecg == "ST-T wave abnormality (ST)":
            features_dict['RestingECG_ST'] = 1
            
        if exercise_angina == "Yes":
            features_dict['ExerciseAngina_Y'] = 1
            
        if st_slope == "Flat":
            features_dict['ST_Slope_Flat'] = 1
        elif st_slope == "Upsloping (Up)":
            features_dict['ST_Slope_Up'] = 1
            
        # Create df and order columns
        df_in = pd.DataFrame([features_dict])[heart_columns]
        
        # Scale
        X_scaled = scaler_heart.transform(df_in)
        
        # Predict
        model_key = model_name.lower().replace(" (rbf kernel)", "").replace(" ", "_")
        model = load_model(f"models/heart_{model_key}.joblib")
        
        if model:
            pred = model.predict(X_scaled)[0]
            
            # Check for probabilities
            prob_str = ""
            prob_val = 0
            has_prob = hasattr(model, "predict_proba")
            
            if has_prob:
                probs = model.predict_proba(X_scaled)[0]
                prob_val = probs[1]
                prob_str = f"Estimated Risk Probability: **{prob_val*100:.1f}%**"
            
            # Color code depending on prediction
            result_color = "#ef4444" if pred == 1 else "#10b981"
            result_bg = "#fef2f2" if pred == 1 else "#ecfdf5"
            result_text = "Heart Disease Detected (High Risk)" if pred == 1 else "Normal (Low Risk)"
            
            st.markdown(f"""
                <div class="glass-card" style="border-left: 6px solid {result_color}; background-color: #ffffff;">
                    <h3 style="color: {result_color}; margin-top: 0;">Evaluation Results</h3>
                    <p style="font-size: 16px;">Based on the patient metrics, the model predicts:</p>
                    <div style="font-size: 24px; font-weight: 700; color: {result_color}; background: {result_bg}; display: inline-block; padding: 10px 20px; border-radius: 8px; border: 1px solid {result_color}; margin-bottom: 15px;">
                        {result_text}
                    </div>
            """, unsafe_allow_html=True)
            
            if has_prob:
                st.markdown(f"<p style='font-size: 16px; margin-bottom: 5px;'>{prob_str}</p>", unsafe_allow_html=True)
                st.progress(float(prob_val))
            
            st.markdown("""
                    <p style="font-size: 13px; color: #64748b; margin-top: 20px;">
                        *Warning: This tool is an educational risk estimator and does not replace medical advice. Always seek advice from a doctor for medical issues.*
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Error loading heart prediction model.")
            
    st.markdown("<div class='custom-hr'></div>", unsafe_allow_html=True)
    
    if metrics_data:
        st.subheader("📊 Model Performance & Evaluation Metrics")
        rows = []
        for name, data in metrics_data.items():
            rows.append({
                "Model Name": name,
                "Accuracy": f"{data['accuracy']*100:.2f}%",
                "Precision": f"{data['precision']*100:.2f}%",
                "Recall": f"{data['recall']*100:.2f}%",
                "F1 Score": f"{data['f1_score']*100:.2f}%",
                "ROC-AUC": f"{data['roc_auc']:.4f}"
            })
        st.table(pd.DataFrame(rows).set_index("Model Name"))
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown(f"**Detailed Confusion Matrix ({model_name})**")
            selected_metrics = metrics_data[model_name]
            cm = np.array(selected_metrics["confusion_matrix"])
            fig_cm = plot_confusion_matrix(cm, ["Normal", "Heart Disease"], "#ef4444")
            st.pyplot(fig_cm)
            
        with col_chart2:
            st.markdown(f"**ROC Curves Comparison (Highlighting: {model_name})**")
            fig_roc = plot_roc_curves(metrics_data, model_name, "#ef4444")
            st.pyplot(fig_roc)


# ------------------------------------------
# PAGE: MENTAL HEALTH RISK
# ------------------------------------------
elif st.session_state.page == "Mental Health":
    apply_theme_css("#3b82f6", "#f0f6ff", "#e0eefe")
    
    render_html_card("""
        <div style="
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            border-radius: 16px;
            padding: 24px;
            color: #ffffff;
            box-shadow: 0 10px 25px rgba(59, 130, 246, 0.15);
            margin-bottom: 30px;
            display: flex;
            align-items: center;
            gap: 20px;
            position: relative;
            overflow: hidden;
            margin-top: 10px;
        ">
            <!-- Decorative background shapes -->
            <div style="position: absolute; right: -30px; top: -30px; width: 120px; height: 120px; border-radius: 50%; background: rgba(255, 255, 255, 0.1); pointer-events: none;"></div>
            <div style="position: absolute; right: 50px; bottom: -50px; width: 100px; height: 100px; border-radius: 50%; background: rgba(255, 255, 255, 0.05); pointer-events: none;"></div>
            
            <!-- Icon Container -->
            <div style="
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 12px;
                padding: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
            ">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="color: white;">
                    <path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/>
                    <path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/>
                    <path d="M12 5v14"/>
                    <path d="M12 12h6"/>
                    <path d="M12 12H6"/>
                </svg>
            </div>
            
            <!-- Text Content -->
            <div style="flex-grow: 1;">
                <h1 style="
                    color: #ffffff !important;
                    margin: 0 0 4px 0 !important;
                    font-size: 24px !important;
                    font-weight: 700 !important;
                    line-height: 1.2 !important;
                    border: none !important;
                    padding: 0 !important;
                ">Mental Health Risk Prediction</h1>
                <p style="
                    color: rgba(255, 255, 255, 0.9) !important;
                    margin: 0 !important;
                    font-size: 14px !important;
                    font-weight: 400 !important;
                    line-height: 1.4 !important;
                ">Assess risk levels (Low, Medium, High) based on life and occupational parameters.</p>
            </div>
        </div>
    """)
    
    # Load resources
    mental_columns = load_model("models/mental_columns.joblib")
    scaler_mental = load_model("models/mental_scaler.joblib")
    mental_encoders = load_model("models/mental_encoders.joblib")
    metrics_data = load_metrics("models/mental_metrics.json")
    
    # Model selection placed prominently below the banner
    model_name = st.selectbox("Select ML Model (applies to both prediction and performance stats below):", list(metrics_data.keys()), key="mental_model_select")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Demographics & Work Profile")
    
    # Categorical forms map options from LabelEncoders
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age:", 18, 90, 35)
        gender = st.selectbox("Gender:", list(mental_encoders['gender'].classes_))
        marital_status = st.selectbox("Marital Status:", list(mental_encoders['marital_status'].classes_))
        education_level = st.selectbox("Education Level:", list(mental_encoders['education_level'].classes_))
        employment_status = st.selectbox("Employment Status:", list(mental_encoders['employment_status'].classes_))
        sleep_hours = st.slider("Sleep Hours (daily average):", 3.0, 12.0, 7.0, step=0.5)
        
    with col2:
        physical_activity = st.slider("Physical Activity Hours/Week:", 0.0, 30.0, 5.0, step=0.5)
        screen_time = st.slider("Screen Time Hours/Day:", 0.0, 24.0, 4.0, step=0.5)
        social_support = st.slider("Social Support Score (1-10):", 1, 10, 5)
        work_stress = st.slider("Work Stress Level (1-10):", 1, 10, 5)
        academic_pressure = st.slider("Academic Pressure Level (1-10):", 1, 10, 5)
        job_satisfaction = st.slider("Job Satisfaction Score (1-10):", 1, 10, 6)
        
    with col3:
        financial_stress = st.slider("Financial Stress Level (1-10):", 1, 10, 5)
        working_hours = st.slider("Working Hours/Week:", 0, 100, 40)
        anxiety_score = st.slider("Anxiety Score (0-21):", 0, 21, 6)
        depression_score = st.slider("Depression Score (0-24):", 0, 24, 6)
        stress_level = st.slider("Stress Level (1-10):", 1, 10, 5)
        mood_swings = st.slider("Mood Swings Frequency (1-10):", 1, 10, 5)
        
    # Extra columns
    col4, col5 = st.columns(2)
    with col4:
        concentration = st.slider("Concentration Difficulty (1-10):", 1, 10, 5)
        panic_history = st.selectbox("Panic Attack History:", ["No", "Yes"])
        family_history = st.selectbox("Family History of Mental Illness:", ["No", "Yes"])
        
    with col5:
        prev_diag = st.selectbox("Previous Mental Health Diagnosis:", ["No", "Yes"])
        therapy_history = st.selectbox("Therapy History:", ["No", "Yes"])
        substance_use = st.selectbox("Substance Use:", ["No", "Yes"])
        
    st.markdown("<hr style='border-color: rgba(0,0,0,0.05);'>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Evaluate Mental Health Risk", type="primary"):
        # Construct dictionary
        input_dict = {}
        
        # Map numericals
        input_dict['age'] = age
        input_dict['sleep_hours'] = sleep_hours
        input_dict['physical_activity_hours_per_week'] = physical_activity
        input_dict['screen_time_hours_per_day'] = screen_time
        input_dict['social_support_score'] = social_support
        input_dict['work_stress_level'] = work_stress
        input_dict['academic_pressure_level'] = academic_pressure
        input_dict['job_satisfaction_score'] = job_satisfaction
        input_dict['financial_stress_level'] = financial_stress
        input_dict['working_hours_per_week'] = working_hours
        input_dict['anxiety_score'] = anxiety_score
        input_dict['depression_score'] = depression_score
        input_dict['stress_level'] = stress_level
        input_dict['mood_swings_frequency'] = mood_swings
        input_dict['concentration_difficulty_level'] = concentration
        
        # Map Categoricals using Encoders
        input_dict['gender'] = mental_encoders['gender'].transform([gender])[0]
        input_dict['marital_status'] = mental_encoders['marital_status'].transform([marital_status])[0]
        input_dict['education_level'] = mental_encoders['education_level'].transform([education_level])[0]
        input_dict['employment_status'] = mental_encoders['employment_status'].transform([employment_status])[0]
        
        # Binaries (No/Yes mapped to 0/1, transformed through encoders if fitting)
        # In the data profile check, they were numeric 0/1. Let's use encoders just in case.
        val_panic = 1 if panic_history == "Yes" else 0
        input_dict['panic_attack_history'] = mental_encoders['panic_attack_history'].transform([val_panic])[0]
        
        val_fam = 1 if family_history == "Yes" else 0
        input_dict['family_history_mental_illness'] = mental_encoders['family_history_mental_illness'].transform([val_fam])[0]
        
        val_diag = 1 if prev_diag == "Yes" else 0
        input_dict['previous_mental_health_diagnosis'] = mental_encoders['previous_mental_health_diagnosis'].transform([val_diag])[0]
        
        val_ther = 1 if therapy_history == "Yes" else 0
        input_dict['therapy_history'] = mental_encoders['therapy_history'].transform([val_ther])[0]
        
        val_sub = 1 if substance_use == "Yes" else 0
        input_dict['substance_use'] = mental_encoders['substance_use'].transform([val_sub])[0]
        
        # Create df
        df_in = pd.DataFrame([input_dict])[mental_columns]
        
        # Scale
        X_scaled = scaler_mental.transform(df_in)
        
        # Predict
        model_key = model_name.lower().replace(" ", "_")
        model = load_model(f"models/mental_{model_key}.joblib")
        
        if model:
            pred = model.predict(X_scaled)[0]
            
            # Check mapping: 0=Low, 1=Medium, 2=High Risk
            risk_labels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
            risk_colors = {0: "#10b981", 1: "#f59e0b", 2: "#ef4444"}
            risk_bgs = {0: "#ecfdf5", 1: "#fefbeb", 2: "#fef2f2"}
            
            risk_text = risk_labels.get(pred, "Unknown")
            risk_color = risk_colors.get(pred, "#6366f1")
            risk_bg = risk_bgs.get(pred, "#f8fafc")
            
            st.markdown(f"""
                <div class="glass-card" style="border-left: 6px solid {risk_color}; background-color: #ffffff;">
                    <h3 style="color: {risk_color}; margin-top: 0;">Risk Analysis Results</h3>
                    <p style="font-size: 16px;">The diagnostic model classifies the mental health risk profile as:</p>
                    <div style="font-size: 24px; font-weight: 700; color: {risk_color}; background: {risk_bg}; display: inline-block; padding: 10px 20px; border-radius: 8px; border: 1px solid {risk_color}; margin-bottom: 15px;">
                        {risk_text}
                    </div>
            """, unsafe_allow_html=True)
            
            # Display probability breakdown if possible
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_scaled)[0]
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    st.metric("Low Risk Probability", f"{probs[0]*100:.1f}%")
                with col_p2:
                    st.metric("Medium Risk Probability", f"{probs[1]*100:.1f}%")
                with col_p3:
                    st.metric("High Risk Probability", f"{probs[2]*100:.1f}%")
                    
            st.markdown("""
                    <p style="font-size: 13px; color: #64748b; margin-top: 20px;">
                        *Note: Mental health evaluation models are screening aids. If you are experiencing distress, please consult a certified psychologist or counsellor.*
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Error loading mental health risk model.")
            
    st.markdown("<div class='custom-hr'></div>", unsafe_allow_html=True)
    
    if metrics_data:
        st.subheader("📊 Model Performance & Evaluation Metrics")
        rows = []
        for name, data in metrics_data.items():
            rows.append({
                "Model Name": name,
                "Accuracy": f"{data['accuracy']*100:.2f}%",
                "Precision": f"{data['precision']*100:.2f}%",
                "Recall": f"{data['recall']*100:.2f}%",
                "F1 Score": f"{data['f1_score']*100:.2f}%"
            })
        st.table(pd.DataFrame(rows).set_index("Model Name"))
        
        st.markdown(f"**Detailed Confusion Matrix ({model_name})**")
        selected_metrics = metrics_data[model_name]
        cm = np.array(selected_metrics["confusion_matrix"])
        fig_cm = plot_confusion_matrix(cm, ["Low", "Medium", "High"], "#3b82f6")
        st.pyplot(fig_cm)


# ------------------------------------------
# PAGE: DIABETES PREDICTION
# ------------------------------------------
elif st.session_state.page == "Diabetes":
    apply_theme_css("#f59e0b", "#fffbeb", "#fef3c7")
    
    render_html_card("""
        <div style="
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            border-radius: 16px;
            padding: 24px;
            color: #ffffff;
            box-shadow: 0 10px 25px rgba(245, 158, 11, 0.15);
            margin-bottom: 30px;
            display: flex;
            align-items: center;
            gap: 20px;
            position: relative;
            overflow: hidden;
            margin-top: 10px;
        ">
            <!-- Decorative background shapes -->
            <div style="position: absolute; right: -30px; top: -30px; width: 120px; height: 120px; border-radius: 50%; background: rgba(255, 255, 255, 0.1); pointer-events: none;"></div>
            <div style="position: absolute; right: 50px; bottom: -50px; width: 100px; height: 100px; border-radius: 50%; background: rgba(255, 255, 255, 0.05); pointer-events: none;"></div>
            
            <!-- Icon Container -->
            <div style="
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 12px;
                padding: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
            ">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="color: white;">
                    <path d="M12 22a7 7 0 0 0 7-7c0-4.3-7-13-7-13S5 10.7 5 15a7 7 0 0 0 7 7z"/>
                </svg>
            </div>
            
            <!-- Text Content -->
            <div style="flex-grow: 1;">
                <h1 style="
                    color: #ffffff !important;
                    margin: 0 0 4px 0 !important;
                    font-size: 24px !important;
                    font-weight: 700 !important;
                    line-height: 1.2 !important;
                    border: none !important;
                    padding: 0 !important;
                ">Diabetes Prediction</h1>
                <p style="
                    color: rgba(255, 255, 255, 0.9) !important;
                    margin: 0 !important;
                    font-size: 14px !important;
                    font-weight: 400 !important;
                    line-height: 1.4 !important;
                ">Assess insulin/glucose clinical records to evaluate diabetes outcome risk.</p>
            </div>
        </div>
    """)
    
    # Load assets
    diab_columns = load_model("models/diabetes_columns.joblib")
    scaler_diab = load_model("models/diabetes_scaler.joblib")
    metrics_data = load_metrics("models/diabetes_metrics.json")
    
    # Model selection placed prominently below the banner
    model_name = st.selectbox("Select ML Model (applies to both prediction and performance stats below):", list(metrics_data.keys()), key="diabetes_model_select")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Physiological Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.slider("Number of Pregnancies:", 0, 17, 3)
        glucose = st.slider("Glucose Level (mg/dl):", 0, 200, 115)
        blood_pressure = st.slider("Blood Pressure (mmHg):", 0, 130, 72)
        skin_thickness = st.slider("Triceps Skin Fold Thickness (mm):", 0, 99, 23)
        
    with col2:
        insulin = st.slider("2-Hour Serum Insulin (mu U/ml):", 0, 846, 100)
        bmi = st.slider("Body Mass Index (BMI):", 0.0, 67.0, 32.0, step=0.1)
        diabetes_pedigree = st.slider("Diabetes Pedigree Function:", 0.08, 2.42, 0.37, step=0.01)
        age = st.slider("Age (Years):", 21, 90, 30)
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("Evaluate Diabetes Risk", type="primary"):
        # Load medians dynamically from source data
        medians = {
            'Glucose': 117.0,
            'BloodPressure': 72.0,
            'SkinThickness': 29.0,
            'Insulin': 125.0,
            'BMI': 32.3
        }
        
        # Input dictionary
        input_dict = {
            'Pregnancies': pregnancies,
            'Glucose': glucose if glucose != 0 else medians['Glucose'],
            'BloodPressure': blood_pressure if blood_pressure != 0 else medians['BloodPressure'],
            'SkinThickness': skin_thickness if skin_thickness != 0 else medians['SkinThickness'],
            'Insulin': insulin if insulin != 0 else medians['Insulin'],
            'BMI': bmi if bmi != 0 else medians['BMI'],
            'DiabetesPedigreeFunction': diabetes_pedigree,
            'Age': age
        }
        
        # Feature engineering
        # 1. AgeGroup
        if age <= 30:
            input_dict['AgeGroup'] = 1
        elif age <= 40:
            input_dict['AgeGroup'] = 2
        elif age <= 50:
            input_dict['AgeGroup'] = 3
        elif age <= 60:
            input_dict['AgeGroup'] = 4
        else:
            input_dict['AgeGroup'] = 5
            
        # 2. BMI_Category
        actual_bmi = input_dict['BMI']
        if actual_bmi < 18.5:
            input_dict['BMI_Category'] = 0
        elif actual_bmi < 25:
            input_dict['BMI_Category'] = 1
        elif actual_bmi < 30:
            input_dict['BMI_Category'] = 2
        else:
            input_dict['BMI_Category'] = 3
            
        # Create df and order columns
        df_in = pd.DataFrame([input_dict])[diab_columns]
        
        # Scale
        X_scaled = scaler_diab.transform(df_in)
        
        # Predict
        model_key = model_name.lower().replace(" ", "_")
        model = load_model(f"models/diabetes_{model_key}.joblib")
        
        if model:
            pred = model.predict(X_scaled)[0]
            
            # Check probabilities
            prob_str = ""
            prob_val = 0
            has_prob = hasattr(model, "predict_proba")
            
            if has_prob:
                probs = model.predict_proba(X_scaled)[0]
                prob_val = probs[1]
                prob_str = f"Estimated Risk Probability: **{prob_val*100:.1f}%**"
                
            # Style outputs
            result_color = "#f59e0b" if pred == 1 else "#10b981"
            result_bg = "#fffbeb" if pred == 1 else "#ecfdf5"
            result_text = "Diabetes Risk Detected (Positive)" if pred == 1 else "Normal (Negative)"
            
            st.markdown(f"""
                <div class="glass-card" style="border-left: 6px solid {result_color}; background-color: #ffffff;">
                    <h3 style="color: {result_color}; margin-top: 0;">Evaluation Results</h3>
                    <p style="font-size: 16px;">Based on physiological measurements, the model predicts:</p>
                    <div style="font-size: 24px; font-weight: 700; color: {result_color}; background: {result_bg}; display: inline-block; padding: 10px 20px; border-radius: 8px; border: 1px solid {result_color}; margin-bottom: 15px;">
                        {result_text}
                    </div>
            """, unsafe_allow_html=True)
            
            if has_prob:
                st.markdown(f"<p style='font-size: 16px; margin-bottom: 5px;'>{prob_str}</p>", unsafe_allow_html=True)
                st.progress(float(prob_val))
                
            st.markdown("""
                    <p style="font-size: 13px; color: #64748b; margin-top: 20px;">
                        *Disclaimer: This evaluation is generated by clinical machine learning screening tools and should not be used as a final diagnostic medical report.*
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Error loading diabetes prediction model.")
            
    st.markdown("<div class='custom-hr'></div>", unsafe_allow_html=True)
    
    if metrics_data:
        st.subheader("📊 Model Performance & Evaluation Metrics")
        rows = []
        for name, data in metrics_data.items():
            rows.append({
                "Model Name": name,
                "Accuracy": f"{data['accuracy']*100:.2f}%",
                "Precision": f"{data['precision']*100:.2f}%",
                "Recall": f"{data['recall']*100:.2f}%",
                "F1 Score": f"{data['f1_score']*100:.2f}%",
                "ROC-AUC": f"{data['roc_auc']:.4f}"
            })
        st.table(pd.DataFrame(rows).set_index("Model Name"))
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown(f"**Detailed Confusion Matrix ({model_name})**")
            selected_metrics = metrics_data[model_name]
            cm = np.array(selected_metrics["confusion_matrix"])
            fig_cm = plot_confusion_matrix(cm, ["Negative", "Positive"], "#f59e0b")
            st.pyplot(fig_cm)
            
        with col_chart2:
            st.markdown(f"**ROC Curves Comparison (Highlighting: {model_name})**")
            fig_roc = plot_roc_curves(metrics_data, model_name, "#f59e0b")
            st.pyplot(fig_roc)
