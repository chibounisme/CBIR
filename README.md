CBIR Project Using VGG-16 and Annoy
====================

Developed by: Dhaoui Alaeddine & Chiboub Mohamed
---------------------

### How to run:

- Clone this repo
- Make sure that you have the the `index/manhattan.ann` index database located on the repository main folder (you can get it from the Faculty's Computer)
- Run the `pip install io streamlit fastapi numpy PIL cv2 annoy tensorflow` command to install the necessary dependencies for the project
- Go to `back_end.py` and change the 'image_server' variable to match with your image server URL
- Run the backend using the `uvicorn back_end:app --reload` command
- Run the frontend using the `streamlit run front_end.py` command
- Open the generated frontend URL and use the CBIR Search-Engine