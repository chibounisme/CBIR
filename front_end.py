import streamlit as st
from io import BytesIO
import requests
import numpy as np
from PIL import Image
import json

backend_url = 'http://127.0.0.1:8000/uploadfile'

def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))

@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img

st.title('CBIR Project')
st.info('Content-Based Image Retrieval using VGG-16 Features')

image = st.file_uploader('Upload your image', type=['jpg', 'jpeg'])
show_file = st.empty()

if not image:
    show_file.info(f'Please upload a file of format: {" - ".join(["PNG", "JPG", "JPEG"])}')

if image and isinstance(image, BytesIO):
    _, search_btn_container, _ = st.columns([1, 1, 1])
    if search_btn_container.button('Search for similar Images'):
        files = {'image': image}
        response = requests.post(backend_url, files=files)
        response = json.loads(response.text)
        time_elapsed = 'Time elapsed: ' + str(response['elapsed_time']) + ' seconds'
        st.text(time_elapsed)
        for image_url in response['img_urls']:
            st.image(image_url)

# to run use: streamlit run front_end.py