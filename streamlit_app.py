import streamlit as st
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm

# ------------------ CSS ------------------
st.markdown("""
<style>
/* MAIN BACKGROUND */
.stApp {
    background: radial-gradient(circle at top,
        #1a2a6c,
        #0f2027,
        #000000);
    color: white;
}

/* TITLE */
.title-text {
    font-size: 3.2rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #4FC3F7, #81D4FA, #4FC3F7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0px 0px 20px rgba(79,195,247,0.6);
}

/* SUBTITLE */
.subtitle-text {
    text-align: center;
    color: #B3E5FC;
    font-size: 1.2rem;
    margin-bottom: 30px;
}

/* IMAGE CARDS */
.image-card {
    background: rgba(15, 32, 39, 0.9);
    border-radius: 20px;
    padding: 10px;
    box-shadow: 0 0 25px rgba(79,195,247,0.4);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    margin-bottom: 15px;
}
.image-card:hover {
    transform: scale(1.05);
    box-shadow: 0 0 35px rgba(79,195,247,0.8);
}

/* SIDEBAR STYLING */
[data-testid="stSidebar"] {
    background-color: rgba(0,0,0,0.8);
    color: white;
    padding: 15px;
    border-radius: 10px;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background-color: rgba(0,0,0,0.4);
    border-radius: 15px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Fashion Recommender",
    page_icon="👗",
    layout="wide"
)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base_model.trainable = False
    return base_model

model = load_model()
feature_extractor = GlobalMaxPooling2D()

# ------------------ LOAD DATA ------------------
embeddings = pickle.load(open('embeddings.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# ------------------ FEATURE EXTRACTION ------------------
def extract_features(img, model):
    img = img.resize((224,224))
    img_array = image.img_to_array(img)
    expanded = np.expand_dims(img_array, axis=0)
    preprocessed = preprocess_input(expanded)
    result = model.predict(preprocessed, verbose=0)
    features = feature_extractor(result).numpy().flatten()
    return features / norm(features)

# ------------------ RECOMMENDATION FUNCTION ------------------
def recommend(features, embeddings, top_n=5):
    similarities = np.dot(embeddings, features)
    indices = np.argsort(similarities)[-top_n:][::-1]
    return indices

# ------------------ SIDEBAR ------------------
st.sidebar.markdown("## ⚙️ Settings & Info")
st.sidebar.markdown("Upload an image to get fashion recommendations")
st.sidebar.markdown("**Tech Used:**")
st.sidebar.markdown("- TensorFlow (ResNet50)")
st.sidebar.markdown("- Streamlit")
st.sidebar.markdown("- Python & NumPy")
st.sidebar.markdown("---")
st.sidebar.markdown("👗 **Fashion Recommender System**")

# ------------------ TITLE ------------------
st.markdown("<div class='title-text'>Fashion Recommender System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>AI-Powered Fashion Similarity Engine</div>", unsafe_allow_html=True)

# ------------------ FILE UPLOADER ------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# ------------------ RECOMMENDATION DISPLAY ------------------
if uploaded_file:
    user_img = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 3])

    # Display user uploaded image
    with col1:
        st.image(user_img, caption="Uploaded Image", use_container_width=True)

    # Extract features & recommend
    features = extract_features(user_img, model)
    indices = recommend(features, np.array(embeddings))

    # Display recommended images as glowing cards
    with col2:
        st.subheader("Recommended Products")
        cols = st.columns(5)
        for i, idx in enumerate(indices):
            with cols[i]:
                st.markdown("<div class='image-card'>", unsafe_allow_html=True)
                st.image(filenames[idx], use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
