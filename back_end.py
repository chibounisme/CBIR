from feature_extractor import FeatureExtractor
from skimage import io
import cv2
from annoy import AnnoyIndex
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
import numpy as np
import time

image_server = 'localhost:3000/static/'

fe = FeatureExtractor()

loaded_index = AnnoyIndex(4096, metric='manhattan')
loaded_index.load('index/manhattan.ann')

dataset_list = open('file_list.txt').read().splitlines()

def get_similar_images_annoy(img_vector, result_count):
    similar_img_ids = loaded_index.get_nns_by_vector(img_vector, result_count, include_distances=True)
    return [dataset_list[i] for i in similar_img_ids[0]], similar_img_ids[1]

app = FastAPI()

def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))

@app.post("/uploadfile")
async def create_upload_file(image: UploadFile = File(...), count: int = 10):
    start = time.time()
    img = cv2.resize(load_image_into_numpy_array(await image.read()), (224, 224))
    features = fe.extract(img)
    img_names, distances = get_similar_images_annoy(features, count)
    img_urls = [f'localhost:3000/static/{img_name}' for img_name in img_names]
    end = time.time()

    return {
        "elapsed_time": end - start,
        "response_count": len(img_names),
        "img_urls": img_urls,
        "distances": distances
    }

# to launch use: uvicorn back_end:app --reload