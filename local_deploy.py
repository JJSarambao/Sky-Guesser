import streamlit as st
import tensorflow as tf
import random as rand


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("MultiWeather.h5")
    return model


model = load_model()

classes = {0: "Cloudy", 1: "Rainy", 2: "Shiny", 3: "Sunrise"}

st.write(
    """
# Sky Guesser - Sky Classifier
### Jesriel John Sarambao - CPE32S6
### Emerging Technologies 2 in CpE
### Dr. Jonathan Taylar"""
)


def say_something(class_val):
    what_to_say = {
        0: [
            "Good time for a walk!",
            "It might rain though. Take care!",
            "You can do anything you want right now.",
            "Have a nice day!",
        ],
        1: [
            "N.. not a good time for a walk",
            "You.. you can go out and play in the rain if you want",
            "Make sure you have your umbrellas today!",
            "Take care! Roads and sidewalks may get slippery at times.",
            "A great day for a hot soup, isn't it?",
        ],
        2: [
            "Shiiiiiiiiny~!",
            "What a nice time to walk outside.",
            "It might get too hot, make sure you're hydrated!",
            "Take an umbrella with you, its for the better",
            "AH-- I'M BURNING-- wait, I'm not a mob from Minecraft.",
            "A great time for a nice cold glass of juice. Or water. Or tea. Anything. Just drink.",
        ],
        3: [
            "Wake up, sleepy head.",
            "A nice new start for the day! Do your best!",
            "YOU'RE GONNA BE LATE! YOU'RE GONNA BE-- nah, I'm just kidding, prepare at your own pace.",
            "Rise and Shine! No, don't do that literally--",
        ]
    }
    return rand.choice(what_to_say[class_val])


file = st.file_uploader("Show me your sky!", type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np

st.balloons()


def import_and_predict(image_data, model):
    size = (128, 128)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction


if file is None:
    st.text("Please upload an image of your sky here")
    st.success("Wow! What a great night sk-- oh, you haven't uploaded anything.")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = classes
    string = f"Hmm.. looks like {class_names[np.argmax(prediction)]} sky to me! {say_something(np.argmax(prediction))}"
    st.success(string)
    st.info(
        "I should have uploaded a YOLOv5 model actually. Thing is.. I have problems making it work outside the YOLOv5 folder :("
    )
