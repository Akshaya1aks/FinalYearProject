import streamlit as st
import pandas as pd
import cv2
from tensorflow.keras.models import load_model
import joblib
import numpy as np 
import PIL

# Load models and data
parkinson_csv = joblib.load('parkinson_csv.C5')
parkinson_gait = joblib.load('gait.C5')
parkinson_spiral = load_model('parkinsons_spiral.h5')
parkinson_mri = load_model('parkinsion_mri.h5')

def predict_parkinson_gait(data):
    test = pd.read_csv(data)
    test.drop(["secim"], axis=1, inplace=True)
    test.drop(["Unnamed: 0"], axis=1, inplace=True)
    pred = parkinson_gait.predict(test)
    # Convert numerical predictions to labels
    pred_labels = ['Parkinson' if pred_val == 1 else 'Normal' for pred_val in pred]
    return pred_labels

def predict_parkinson_csv(data):
    test = pd.read_csv(data)
    test.drop(["status"], axis=1, inplace=True)
    test.drop(["name"], axis=1, inplace=True)
    test.drop(["Unnamed: 0"], axis=1, inplace=True)
    pred = parkinson_csv.predict(test)
    # Convert numerical predictions to labels
    pred_labels = ['Parkinson' if pred_val == 1 else 'Normal' for pred_val in pred]
    return pred_labels

def predict_parkinson_mri(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = img / 255.0
    img = cv2.resize(img, (128, 128))
    img = img.reshape(1, 128, 128, 3)
    prediction = parkinson_mri.predict(img)
    # Convert numerical prediction to label
    pred_label = 'Parkinson💉' if prediction[0][0] > 0.5 else 'Normal🧬'
    return pred_label

def predict_parkinson_spiral(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = img / 255.0
    img = cv2.resize(img, (128, 128))
    img = img.reshape(1, 128, 128, 3)
    prediction = parkinson_spiral.predict(img)
    # Convert numerical prediction to label
    pred_label = 'Parkinson💉' if prediction[0][0] > 0.5 else 'Normal🧬'
    return pred_label


# Streamlit App
st.title("Parkinson's Detection App")
st.sidebar.subheader("Parkinson's Disease")
st.sidebar.image('old-1015484_1920.jpg')
st.sidebar.markdown("""
Parkinson's disease is a neurodegenerative disorder that affects movement. 
It develops gradually, sometimes starting with a barely noticeable tremor in just one hand. 
While tremors are common, the disorder also commonly causes stiffness or slowing of movement.
\n\n
Parkinson's disease progresses over time and has no cure, but treatment can help manage symptoms.
""")

# Lists to store individual predictions
individual_results = []

# Function to predict overall result
def predict_overall_result():
    if individual_results:
        overall_result = "Normal - No Parkinson's disease detected" if all(result == 'Normal' for result in individual_results) else "Parkinson's disease detected - Person is unhealthy"
        st.write("Overall Result:", overall_result)
        # Provide recommendation if Parkinson's disease is detected
        if "Parkinson" in overall_result:
            st.write("Please consult a healthcare professional for further evaluation.")
    else:
        st.warning("Please upload data or images first to predict individual results.")

# CSV File Upload for Gait Data
gait_csv_file = st.file_uploader("Choose a CSV file for Gait Data", type=["csv"])
if gait_csv_file is not None:
    prediction_result_gait = predict_parkinson_gait(gait_csv_file)
    st.write("Prediction Result for Gait Data:", prediction_result_gait)
    individual_results.extend(prediction_result_gait)

# CSV File Upload for Spiral Data
spiral_csv_file = st.file_uploader("Choose a CSV file for Spiral Data", type=["csv"])
if spiral_csv_file is not None:
    prediction_result_spiral = predict_parkinson_csv(spiral_csv_file)
    st.write("Prediction Result for Spiral Data:", prediction_result_spiral)
    individual_results.extend(prediction_result_spiral)

# Image File Upload for Spiral Image
spiral_image_file = st.file_uploader("Choose an Image file for Spiral")
if spiral_image_file is not None:
    prediction_result_spiral_image = predict_parkinson_spiral(spiral_image_file)
    st.write("Prediction Result for Spiral Image:", prediction_result_spiral_image)
    individual_results.extend(prediction_result_spiral_image)

# Image File Upload for MRI Image
mri_image_file = st.file_uploader("Choose an Image file for MRI")
if mri_image_file is not None:
    prediction_result_mri_image = predict_parkinson_mri(mri_image_file)
    st.write("Prediction Result for MRI Image:", prediction_result_mri_image)
    individual_results.extend(prediction_result_mri_image)

# Button to predict overall result
if st.button("Predict Overall Result"):
    st.image('clipboard-293_512.gif')
    predict_overall_result()
