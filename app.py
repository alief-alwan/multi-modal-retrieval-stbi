import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import torch
import open_clip
import faiss
import numpy as np
import logging
from typing import List, Tuple, Optional
from torchvision import transforms
from flask import Flask, request, render_template, redirect, send_from_directory, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Command line arguments simulation (replace with actual paths)
text_data_path = 'models/styles.csv'
image_data_path = 'models/images.csv'
text_embeddings_path = 'models/text_embeddings.npy'
image_embeddings_path = 'models/image_embeddings.npy'
batch_size = 32


### Step 1: Load Data from CSV Files ###
def load_data(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {str(e)}")
        raise


### Preprocess Text and Images ###
def preprocess_text(text: str) -> str:
    text = text.lower().strip().split()
    return ' '.join(text)


image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    return image_transforms(image)


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def download_image(url: str, save_path: str) -> Optional[Image.Image]:
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img.save(save_path)  # Save the image locally
        return img
    except Exception as e:
        logging.error(f"Error downloading image from {url}: {str(e)}")
        return None


### Initialize DataFrames ###
text_df = load_data(text_data_path)
image_df = load_data(image_data_path)

# Data Preprocessing
text_df['id'] = text_df['id'].astype(str)
image_df['filename'] = image_df['filename'].str.replace('.jpg', '')

merged_df = pd.merge(text_df, image_df, left_on='id', right_on='filename')

# Concatenate and preprocess relevant columns to form comprehensive descriptions
merged_df['combined'] = merged_df.apply(
    lambda row: ' '.join([
        str(row['gender']),
        str(row['masterCategory']),
        str(row['subCategory']),
        str(row['articleType']),
        str(row['baseColour']),
        str(row['season']),
        str(row['year']),
        str(row['usage']),
        str(row['productDisplayName'])
    ]), axis=1)

product_descriptions = [preprocess_text(desc) for desc in merged_df['combined'].tolist()]
image_urls = merged_df['link'].tolist()

logging.info(f"Loaded {len(product_descriptions)} product descriptions and {len(image_urls)} image URLs.")

# Download and Preprocess Images with Cache Mechanism
image_cache = {}


def get_image(url: str) -> Optional[torch.Tensor]:
    img_filename = os.path.join('models/img', os.path.basename(url))
    if os.path.exists(img_filename):  # Check if the image already exists locally
        img = Image.open(img_filename).convert('RGB')
    else:
        img = download_image(url, img_filename)

    if img is not None:
        preprocessed_img = preprocess_image(img)
        image_cache[url] = preprocessed_img
        return preprocessed_img
    else:
        return None


image_list = [get_image(url) for url in image_urls]
image_list = [img for img in image_list if img is not None]

# Load OpenCLIP Model and Processor
model_name = "hf-hub:Marqo/marqo-fashionCLIP"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, processor = open_clip.create_model_and_transforms(model_name)
model = model.to(device)


### Extract Features in Batches ###
def extract_text_features(texts: List[str], batch_size: int) -> np.ndarray:
    all_features = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = open_clip.tokenize(batch_texts).to(device)
        with torch.no_grad():
            outputs = model.encode_text(inputs)
        all_features.append(outputs.cpu().numpy())
    return np.vstack(all_features)


def extract_image_features(images: List[torch.Tensor], batch_size: int) -> np.ndarray:
    all_features = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        inputs = torch.stack(batch_images).to(device)
        with torch.no_grad():
            outputs = model.encode_image(inputs)
        all_features.append(outputs.cpu().numpy())
    return np.vstack(all_features)


# Save or Load Embeddings
if os.path.exists(text_embeddings_path) and os.path.exists(image_embeddings_path):
    text_features = np.load(text_embeddings_path)
    image_features = np.load(image_embeddings_path)
    logging.info("Loaded embeddings from file.")
else:
    text_features = extract_text_features(product_descriptions, batch_size)
    image_features = extract_image_features(image_list, batch_size)
    np.save(text_embeddings_path, text_features)
    np.save(image_embeddings_path, image_features)
    logging.info("Extracted and saved embeddings.")


### Initialize FAISS Index ###
def init_faiss_index(d: int) -> faiss.IndexFlatL2:
    return faiss.IndexFlatL2(d)


faiss_index = init_faiss_index(image_features.shape[1])
faiss_index.add(image_features)


### Search FAISS Index ###
def search_index(index: faiss.IndexFlatL2, query_features: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
    distances, indices = index.search(query_features, k)
    return list(zip(indices[0], distances[0]))


### Retrieve Similars ###
def retrieve_similar_texts(text_query: str, k: int = 5) -> List[Tuple[int, float]]:
    query_features = extract_text_features([preprocess_text(text_query)], batch_size)
    return search_index(faiss_index, query_features, k)


def retrieve_similar_images(query_image_url: str, k: int = 5) -> List[Tuple[int, float]]:
    img = get_image(query_image_url)
    if img is None:
        return []

    query_image_features = extract_image_features([img], batch_size)
    return search_index(faiss_index, query_image_features, k)


def retrieve_images_by_text(text_query: str, k: int = 5) -> List[Tuple[int, float]]:
    query_features = extract_text_features([preprocess_text(text_query)], batch_size)
    return search_index(faiss_index, query_features, k)


def retrieve_similar_uploaded_image(uploaded_img: Image.Image, k: int = 5) -> List[Tuple[int, float]]:
    preprocessed_img = preprocess_image(uploaded_img)
    query_image_features = extract_image_features([preprocessed_img], batch_size)
    return search_index(faiss_index, query_image_features, k)


# Routes
@app.route('/')
def homepage():
    return render_template('index.html')


@app.route('/search_by_text', methods=['POST'])
def search_by_text():
    query_text = request.form['query_text']
    similar_products_by_text = retrieve_similar_texts(query_text)
    similar_images_by_text = retrieve_images_by_text(query_text)
    indices_and_scores = [(idx, score) for (idx, score) in
                          similar_images_by_text]  # Indices and scores for similar images
    return render_template('search_results.html',
                           text_query=query_text,
                           similar_products_by_text=[product_descriptions[idx] for idx, _ in similar_products_by_text],
                           image_urls=image_urls,  # Pass image URLs list
                           indices_and_scores=indices_and_scores)  # Pass indices and scores


@app.route('/search_by_image', methods=['POST'])
def search_by_image():
    query_image_url = request.form['query_image_url']
    similar_products_by_image = retrieve_similar_images(query_image_url)
    indices_and_scores = [(idx, score) for (idx, score) in
                          similar_products_by_image]  # Indices and scores for similar images
    return render_template('search_results.html',
                           image_url=query_image_url,
                           image_urls=image_urls,  # Pass image URLs list
                           indices_and_scores=indices_and_scores)  # Pass indices and scores


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/search_by_image_upload', methods=['POST'])
def search_by_image_upload():
    if 'query_image_file' not in request.files:
        return redirect(request.url)

    file = request.files['query_image_file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        uploaded_img = Image.open(filepath).convert('RGB')
        similar_images_by_upload = retrieve_similar_uploaded_image(uploaded_img)
        indices_and_scores = [(idx, score) for (idx, score) in similar_images_by_upload]

        return render_template('search_results.html',
                               image_file_url=url_for('uploaded_file', filename=filename),
                               image_urls=image_urls,
                               indices_and_scores=indices_and_scores)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
