import cv2
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image

model = tf.keras.models.load_model('animal-10.hdf5')
classes = ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']
translate = ["pas", "konj", "slon", "leptir", "kokoš", "mačka", "krava", "ovca", "pauk", "vjeverica"]

st.title("Klasifikacija životinja")
upload = st.file_uploader(label='Učitaj sliku:')

if upload is not None:
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    img = Image.open(upload)
    st.image(img, width=300)
    if st.button('Klasificiraj'):
        x = cv2.resize(opencv_image, (224, 224))
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.vgg16.preprocess_input(x)
        result = model.predict(x)
        prediction = translate[np.argmax(result[0])]
        pred = classes[np.argmax(result[0])]
        st.title(f'Na slici se nalazi {prediction}')
