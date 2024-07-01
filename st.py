## Written by Sujit Haloi
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import cv2  # OpenCV for image processing

# Load the image classification model (assuming it's a TensorFlow model)
model = tf.keras.models.load_model("project_01_02.keras")

def welcome():
    return "Welcome All"

def preprocess_image(image):
    """
    Preprocess the image to the required input shape and scale.
    """
    image = np.array(image)  # Convert PIL Image to numpy array
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    img = cv2.resize(img, (256, 256))  # Resize to 256x256
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_image_authentication(image):
    """Let's Authenticate the Image
    This function takes an image as input and returns the prediction
    whether the image is real or AI-generated.
    """
    # Preprocess the image to the required input shape and scale
    img_array = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(img_array)
    
    confidence = prediction[0][0] * 100
    
    if confidence <= 50:
        result = f"Provided image is most probably AI-generated. Model is {100 - confidence:.2f}% sure."
    else:
        result = f"Provided image is most probably NOT AI-generated. Model is {confidence:.2f}% sure."
    
    return result

def main():
    st.title("AI Image Classifier.")
    html_temp = """
    <div style="background-color:rgb(255, 153, 0);padding:10px">
    <h2 style="color:white;text-align:center;">IS IT AI or NOT? </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Pick an image...[Image size limit: 10MB]", type=["jpg", "jpeg", "png", "bmp"])
    result = ""
    if uploaded_file is not None:
        file_size = uploaded_file.size
        if file_size > 10 * 1024 * 1024:
            st.error("File size should be less than 10MB")
        else:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            if st.button("Predict"):
                st.write("Classifying...")
                result = predict_image_authentication(image)
                st.success('Result :: {}'.format(result))
                st.markdown('[CLICK, if not satisfied with the result.](https://forms.gle/iy1e8Bv1urd5JJCA6)', unsafe_allow_html=True)
    
    # Adding the dark green block with "Thank You"
    st.markdown("""
    <div style="background-color:darkgreen;padding:10px;margin-top:20px">
    <h3 style="color:white;text-align:center;">Thank You</h3>
    Developed By: Sandesh, Sahil, Tarun K Ch, Sujit H.
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
