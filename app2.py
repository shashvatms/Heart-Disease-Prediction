# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
import os

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

# Set Streamlit page configuration
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# Custom CSS for background image and fonts

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url('https://images.unsplash.com/photo-1588776814546-b74b66f4838d?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1588776814546-b74b66f4838d?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-attachment: fixed;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border: None;
        border-radius: 10px;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #ff0000;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main title and subtitle
language = st.selectbox("Choose your language", ("English", "Hindi"))

# Title and subtitle based on language
if language == "English":
    st.title("❤️ Heart Disease Prediction")
    st.subheader("🔎 Analyze your health and predict heart disease risk instantly!")
    advice_header = "💬 Health Advice:"
    risk_message = "Risk Assessment:"
    prediction_message = "Prediction Result:"
    disclaimer = "Model is only for educational purposes, not medical advice."
else:
    st.title("❤️ हार्ट डिजीज प्रेडिक्शन")
    st.subheader("🔎 अपनी सेहत का विश्लेषण करें और दिल की बीमारी के खतरे का अनुमान लगाएं!")
    advice_header = "💬 स्वास्थ्य सलाह:"
    risk_message = "जोखिम मूल्यांकन:"
    prediction_message = "भविष्यवाणी परिणाम:"
    disclaimer = "मॉडल केवल शैक्षिक उद्देश्यों के लिए है, यह चिकित्सा सलाह नहीं है।"

# File uploader
uploaded_file = st.sidebar.file_uploader("📄 Upload your medical report (CSV only)", type=["csv"])

def preprocess_uploaded_file(file):
    df = pd.read_csv(file)
    required_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    if all(col in df.columns for col in required_columns):
        return df[required_columns].iloc[0:1]  # Take first row
    else:
        st.error("Uploaded file does not match required format!")
        return None


def user_input_features():
    st.sidebar.markdown("### 📋 Fill Patient Details Below:")

    # --- Age
    age = st.sidebar.slider('Age (Years)', 18, 100, 35, help="Patient's age in years.")

    # --- Sex
    sex = st.sidebar.selectbox('Sex', ['Male', 'Female'], help="Biological gender.")

    # --- Chest Pain
    cp_options = { 'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3 }
    cp = st.sidebar.selectbox('Chest Pain Type', list(cp_options.keys()), help="Type of chest pain experienced.")
    if cp == 'Asymptomatic':
        st.sidebar.warning("⚠️ Asymptomatic chest pain can be serious. Please consult a doctor.")

    # --- Resting Blood Pressure
    trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 80, 200, 120,
                                 help="Normal BP ~120 mm Hg.")
    if trestbps > 140:
        st.sidebar.error("🔴 High blood pressure detected!")
    elif trestbps < 90:
        st.sidebar.warning("🟠 Low blood pressure detected.")

    # --- Serum Cholesterol
    chol = st.sidebar.slider('Serum Cholesterol (mg/dl)', 100, 400, 180,
                              help="Normal cholesterol ~180-200 mg/dl.")
    if chol > 240:
        st.sidebar.error("🔴 High cholesterol! Risk of heart disease.")
    elif chol < 200:
        st.sidebar.success("🟢 Cholesterol in healthy range.")

    # --- Fasting Blood Sugar
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'],
                               help="Is fasting blood sugar higher than 120?")
    if fbs == 'Yes':
        st.sidebar.warning("⚠️ High fasting sugar detected. Diabetes risk.")

    # --- Resting ECG
    restecg_options = { 'Normal': 0, 'ST-T Wave Abnormality': 1, 'Left Ventricular Hypertrophy': 2 }
    restecg = st.sidebar.selectbox('Resting ECG Result', list(restecg_options.keys()),
                                   help="Results from resting ECG test.")

    # --- Max Heart Rate
    thalach = st.sidebar.slider('Max Heart Rate Achieved', 60, 220, 170,
                                 help="Normal max HR ~170-190 during exercise.")
    if thalach < 100:
        st.sidebar.warning("🟠 Low heart rate. May indicate cardiac issues.")

    # --- Exercise Induced Angina
    exang = st.sidebar.selectbox('Exercise Induced Angina', ['No', 'Yes'],
                                 help="Chest pain during exercise?")
    if exang == 'Yes':
        st.sidebar.warning("⚠️ Chest pain during exercise detected.")

    # --- Oldpeak
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise', 0.0, 6.0, 1.0,
                                help="Amount of ST depression.")
    if oldpeak > 2.0:
        st.sidebar.warning("⚠️ Significant ST depression observed.")

    # --- Slope
    slope_options = { 'Upsloping': 0, 'Flat': 1, 'Downsloping': 2 }
    slope = st.sidebar.selectbox('Slope of Peak Exercise ST Segment', list(slope_options.keys()),
                                 help="Slope change after exercise.")
    if slope == 'Downsloping':
        st.sidebar.warning("⚠️ Downsloping ST segment can indicate heart issues.")

    # --- Major vessels
    ca = st.sidebar.selectbox('Number of Major Vessels Colored (0-3)', [0, 1, 2, 3],
                              help="Number of vessels visible by fluoroscopy.")

    # --- Thalassemia
    thal_options = { 'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3 }
    thal = st.sidebar.selectbox('Thalassemia', list(thal_options.keys()),
                                help="Blood disorder affecting RBCs.")
    if thal != 'Normal':
        st.sidebar.warning("⚠️ Thalassemia-related defect detected.")

    # Mapping
    sex = 1 if sex == 'Male' else 0
    cp = cp_options[cp]
    fbs = 1 if fbs == 'Yes' else 0
    restecg = restecg_options[restecg]
    exang = 1 if exang == 'Yes' else 0
    slope = slope_options[slope]
    thal = thal_options[thal]

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features


# Load input
if uploaded_file:
    input_df = preprocess_uploaded_file(uploaded_file)
else:
    input_df = user_input_features()

# Button for Prediction
# Predict button


if st.button('🚀 Predict'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Prediction Result
    st.subheader(prediction_message)
    if prediction[0] == 0:
        st.error('🚨 High Risk of Heart Disease!')
        st.snow()  # ❄️ Animation
    else:
        st.success('✅ No significant risk detected.')
        st.balloons()  # 🎈 Animation

    # Risk Pie Chart
    st.subheader(risk_message)

    risk = prediction_proba[0][0] * 100
    safe = 100 - risk

    fig, ax = plt.subplots()
    ax.pie([safe, risk], labels=['Safe', 'Risk'], colors=['#00C853', '#D50000'],
           autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    # Real-time Health Advice
    st.subheader(advice_header)
    if prediction[0] == 0:
        st.write("⚠️ **Advice**: Please consult a cardiologist immediately. Adopt a healthy diet, exercise regularly, and avoid smoking/alcohol.")
    else:
        st.write("🎉 **Advice**: Great job! Keep eating healthy, exercising, and schedule annual checkups.")
        
def generate_pdf(input_data, prediction):
    # Create a PDF object
    pdf = FPDF()

    # Add a page
    pdf.add_page()

    # Set font
    pdf.set_font('Arial', 'B', 12)

    # Add title
    pdf.cell(200, 10, txt="Patient Health Report", ln=True, align='C')

    # Add patient details
    pdf.ln(10)  # Line break
    pdf.cell(200, 10, txt="Patient Details:", ln=True)
    
    # Loop through the input_data dictionary
    for key, value in input_data.items():
        if isinstance(value, float) or isinstance(value, int):
            value = f"{value:.2f}"  # Format float to two decimal places
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    # Add model prediction and health advice
    pdf.ln(10)  # Line break
    if prediction == 1:
        pdf.cell(200, 10, txt="Prediction: Healthy (No Heart Disease)", ln=True)
        pdf.cell(200, 10, txt="Health Advice: Keep following a balanced diet and exercise regularly.", ln=True)
    else:
        pdf.cell(200, 10, txt="Prediction: High Risk of Heart Disease", ln=True)
        pdf.cell(200, 10, txt="Health Advice: Please consult a healthcare professional immediately.", ln=True)

    # Save the PDF to a file
    file_path = "patient_report.pdf"
    pdf.output(file_path)

    # Return the file path
    return file_path

if st.button('Download Patient Report as PDF'):
    # Get user input

    # Make a prediction
    prediction = model.predict(input_df)[0]  # Make the prediction and get the value

    # Generate the PDF with prediction and health advice
    file_path = generate_pdf(input_df.iloc[0].to_dict(), prediction)  # Pass both data and prediction

    # Provide the option to download the file
    with open(file_path, "rb") as f:
        st.download_button(
            label="Click here to download your report",
            data=f,
            file_name="patient_report.pdf",
            mime="application/pdf"
        )
    
    # Optional: Remove the file after download (clean up)
    os.remove(file_path) #an up the file after download
# Expandable section to see user input
with st.expander("🗒️ View Entered Details"):
    st.write(input_df)

# Disclaimer
st.markdown(f"**⚠️ Disclaimer:** {disclaimer}")

# Footer
st.markdown(
    """
    <hr style="margin-top:2rem;">
    <center>
    Made with ❤️ by Shashvat | 2025
    </center>
    """,
    unsafe_allow_html=True
)