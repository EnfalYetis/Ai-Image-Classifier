import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import(
MobileNetV2, preprocess_input, decode_predictions
)
from PIL import Image

def load_model():
    model = MobileNetV2(weights="imagenet")
    return model


def preprocess_image(image):
    img=np.array(image)
    img=cv2.resize(img,(224,224))
    img=preprocess_input(img)
    img=np.expand_dims(img,axis=0)
    return img

def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return None

def main():
    st.set_page_config(page_title="Image Classifier", page_icon="👽", layout="centered")
    st.title("AI Image Classifier")
    st.markdown(
    """
    <style>
    /* Genel arka plan */
    .stApp {
        background-color: #FDE2E4;
        color: #333333;
    }

    /* File uploader box (drop area) */
    .stFileUploader > div:first-child {
        background-color: #E0BBE4 !important; /* pastel mor */
        color: #FFFFFF !important; /* yazı rengi */
        border-radius: 12px !important;
        padding: 20px !important;
        border: 2px dashed #CBA0DC !important;
    }


    /* Browse files butonu */
    .stFileUploader button {
        background-color: #CBA0DC !important; /* pastel mor */
        color: #FFFFFF !important; /* yazı rengi */
        border-radius: 10px !important;
        padding: 5px 15px !important;
        font-weight: bold !important;
        border: none !important;
    }

    div[data-baseweb="file-uploader"] button:hover {
        background-color: #D8B7EB !important; /* hover rengi */
        transform: scale(1.05);
    }

    /* Buton stili */
    .stButton>button {
        background: linear-gradient(to right, #FFB6C1, #FFC0CB);
        color: #ffffff;
        border-radius: 15px;
        height: 50px;
        width: 200px;
        font-size: 18px;
        font-weight: bold;
        border: 2px solid #FF69B4;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background: linear-gradient(to right, #FFC0CB, #FFB6C1);
        transform: scale(1.05);
    }

    /* Başlık */
    h1 {
        color: #FF6F91; 
        font-family: "Comic Sans MS", cursive, sans-serif;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #FFF0F5;
        border-radius: 15px;
        padding: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


    st.write("Upload an image and let the AI classify it!")
    tab1, tab2 = st.tabs(["Upload", "About"])
    with tab1:
        st.write("Burada görsel yükleme olacak.")
    with tab2:
        st.write("Proje hakkında bilgi.")
    


    @st.cache_resource
    def load_cache_model():
        return load_model() 

    model = load_cache_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
   
    if uploaded_file is not None:
        image = st.image(
            uploaded_file, caption='Uploaded Image.', use_container_width=True
        )
        btn = st.button("Classify Image")
        if btn:
            with st.spinner('Classifying...'):
                image = Image.open(uploaded_file)
                predictions = classify_image(model, image)
                if predictions:
                    st.subheader("Predictions:")
                    for _, label, score in predictions:
                        st.markdown(f"<p style='color:#FF69B4; font-weight:bold;'>{label}: {score:.2%}</p>", unsafe_allow_html=True)

                    label_name = predictions[0][1]  # en yüksek tahmin


if __name__ == "__main__":
    main()