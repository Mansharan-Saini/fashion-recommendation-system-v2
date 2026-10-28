# Fashion Recommender System

An image-based fashion recommendation app built with Streamlit and TensorFlow. Upload a clothing image to find five visually similar products from the included image collection.

The app uses a pretrained ResNet50 model to extract image features, then ranks the catalog by feature-vector similarity.

## Features

- Upload a JPG, JPEG, or PNG image.
- Generate image features with ResNet50 pretrained on ImageNet.
- Display the five closest matches from the product image collection.
- View recommendations in a Streamlit web interface.

## Project files

- `streamlit_app.py` — Streamlit interface and recommendation logic.
- `app.py` — generates catalog embeddings from the images.
- `images/` — product images used for recommendations.
- `embeddings.pkl` — precomputed product feature vectors (stored with Git LFS).
- `filenames.pkl` — image paths corresponding to the feature vectors.

## Requirements

- Python with a TensorFlow-compatible version
- Git
- Git LFS (required to download `embeddings.pkl`)

## Setup and run

Install Git LFS and clone the repository:

```powershell
git lfs install
git clone https://github.com/Mansharan-Saini/fashion-recommendation-system-v2.git
cd fashion-recommendation-system-v2
```

Install the app dependencies and start Streamlit:

```powershell
python -m pip install streamlit tensorflow numpy pillow
python -m streamlit run streamlit_app.py
```

Streamlit prints a local URL when the app starts. Open it in your browser and upload a clothing image to see recommendations. The first run may take longer while TensorFlow downloads the pretrained ResNet50 weights.

## Regenerate product embeddings

To rebuild the catalog features after changing the product images, install `tqdm` and run the preprocessing script from the project root:

```powershell
python -m pip install tqdm
python app.py
```

This processes every image in `images/` and rewrites `embeddings.pkl` and `filenames.pkl`. Processing the full collection can take some time. The app expects the image paths in `filenames.pkl` to be valid relative to the project root.
