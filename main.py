import os
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from chat_bot import chat


# ------------------------------------------------------------------------------------
# Streamlit Page Configuration
st.set_page_config(
    page_title="Lung Cancer Classification 🫁",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container {
  padding-top: 0.5rem;
}
.sidebar .sidebar-header {
  position: absolute;
  top: 0;
  width: 100%;
  background-color: #fff;
  z-index: 1;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------------
# MyModel Class to handle model-related tasks
class MyModel:
    def __init__(self, model_path):
        self.model = self.load_saved_model(model_path)

    def load_saved_model(self, model_path):
        """
        Load the trained lung cancer classification model.
        """
        model = tf.keras.models.load_model(model_path)
        return model

    def preprocess_image(self, image, target_size=(350, 350)):
        """
        Preprocess the input image for prediction.
        - Resize the image.
        - Normalize pixel values.
        - Expand dimensions for model input.
        """
        if image is not None:
            image = image.resize(target_size)
            image = np.array(image) / 255.0  # Normalize to [0, 1]
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            return image
        return None

    def predict(self, image, class_labels):
        """
        Predict the class of the uploaded image.
        """
        if image is not None:
            predictions = self.model.predict(image)
            predicted_index = np.argmax(predictions)
            confidence = predictions[0][predicted_index]
            predicted_label = class_labels[predicted_index]
            return predicted_label, confidence * 100
        return "No Image", 0.0


# ------------------------------------------------------------------------------------
# UI Class for managing Streamlit UI
class Ui:
    def __init__(self, class_labels):
        self.class_labels = class_labels
        st.title("🫁 Lung Cancer Classification 🩺")
        st.sidebar.header("Upload CT Scan Image")

    def load_image(self):
        """
        Allow user to upload a CT scan image via the sidebar.
        """
        uploaded_file = st.sidebar.file_uploader("Choose a CT scan image (PNG/JPG):", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            if image.mode == 'RGBA':  # Convert RGBA to RGB
                image = image.convert('RGB')
            st.sidebar.image(image, caption="Uploaded Image")
            return image
        return None

    def show_result(self, label, confidence):
        """
        Display the prediction result and confidence in the sidebar.
        """
        st.sidebar.info(f"**Predicted Class:** {label}")
        st.sidebar.info(f"**Accuracy :** {confidence:.2f}%")
        if confidence > 90:
            st.balloons()

    def chat_bot_interface(self):
        chat()

    # def show_prediction_image(self, image, label, confidence):
    #     """
    #     Display the prediction result directly on the main screen.
    #     """
    #     if image is not None:
    #         st.image(image, caption=f"Prediction: {label} | Accuracy : {confidence:.2f}%")


# ------------------------------------------------------------------------------------
# Main Functionality
if __name__ == "__main__":
    # Define model path and class labels
    MODEL_PATH = "trained_lung_cancer_model.h5"
    CLASS_LABELS = ['Normal', 'Adenocarcinoma', 'Large Cell Carcinoma', 'Squamous Cell Carcinoma']

    # Initialize the model and UI classes
    model_obj = MyModel(MODEL_PATH)
    ui = Ui(CLASS_LABELS)

    # Load the image from the user
    image = ui.load_image()

    # Predict the image class if uploaded
    if image is not None:
        processed_image = model_obj.preprocess_image(image)
        label, confidence = model_obj.predict(processed_image, CLASS_LABELS)

        if label == 'Normal':
            result_label = 'Affected'
        else:
            result_label = 'Normal'

        # Display the results
        ui.show_result(result_label, confidence)
        ui.chat_bot_interface()

    # Footer
    st.sidebar.write("Powered by Tajdar")
    st.write("Department of Computer Science")
