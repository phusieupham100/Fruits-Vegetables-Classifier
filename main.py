import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import load_model
import requests
from bs4 import BeautifulSoup

# Load the model
model = load_model('Fruits&VegetablesModel.h5')

# Labels dictionary
labels = {
    0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum',
    6: 'carrot', 7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant',
    12: 'garlic', 13: 'ginger', 14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
    19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
    26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
    32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'
}

# Fruits and vegetables categories
fruits = ['Apple, Banana, Pear, Grapes, Orange, Kiwi, Watermelon, Pomegranate, Pineapple, Mango']
vegetables = [
    "Cucumber", "Carrot", "Capsicum", "Onion", "Potato", "Lemon", "Tomato", "Raddish",
    "Beetroot", "Cabbage", "Lettuce", "Spinach", "Soy Bean", "Cauliflower", "Bell Pepper",
    "Chilli Pepper", "Turnip", "Corn", "Sweetcorn", "Sweet Potato", "Paprika", "Jalepeño",
    "Ginger", "Garlic", "Peas", "Eggplant"
]

# Function to fetch calories
def fetch_calories(prediction):
    try:
        url = 'https://www.google.com/search?&q=calories in ' + prediction
        req = requests.get(url).text
        scrap = BeautifulSoup(req, 'html.parser')
        calories = scrap.find('div', class_='BNeawe iBp4i AP7Wnd').text
        return calories
    except Exception as e:
        st.error("Sorry! Calories not found.")
        print(e)

# Function to process the uploaded image
def processed_img(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = int(" ".join(str(x) for x in y_class))
    res = labels[y]
    return res.capitalize()

# Function to run the app
def run():
    # Page title and layout
    st.set_page_config(page_title="Fruits & Vegetables Classifier", layout="wide")
    st.title(" HUST 🍎 Fruits & 🥦 Vegetables Classifier")
    st.markdown("""
        Welcome to the **Fruits & Vegetables Classifier**!  
        Upload an image of a fruit or vegetable, and this app will predict the category 
        along with its estimated calories.
    """)

    # Sidebar for additional options
    st.sidebar.header("App Options")
    st.sidebar.markdown("""
        - **Upload an image**: Choose a fruit or vegetable image.  
        - **Prediction**: See the result and calories.  
        - **About**: Learn more about the app.
    """)

    # File uploader
    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if img_file is not None:
        # Display the uploaded image
        st.image(img_file, caption="Uploaded Image", use_column_width=True)
        save_image_path = './upload_images/' + img_file.name
        with open(save_image_path, 'wb') as f:
            f.write(img_file.getbuffer())

        # Prediction and result display
        if img_file is not None:
            result = processed_img(save_image_path)
            if result in vegetables:
                st.info('**Category**: 🥦 Vegetables')
            else:
                st.info('**Category**: 🍎 Fruits')
            st.success("**Prediction**: " + result)
            
            # Fetch calories
            cal = fetch_calories(result)
            if cal:
                st.warning('Calories: **' + cal + '**')

    # Footer section
    st.markdown("""
    ---
    _Developed with ❤️ by [Phùng Hữu Phú](https://github.com/phusieupham100)._  
    """)

run()
