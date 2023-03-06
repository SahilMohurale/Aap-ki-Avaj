import cv2
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyautogui
import pyttsx3

engine = pyttsx3.init()  # Windows
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
rate = engine.getProperty('rate')
engine.setProperty('rate', rate - 10)


pyautogui.FAILSAFE = False

st.set_page_config(page_title="Hand Gesture to Speech Conversion", page_icon=":speaker:", layout="wide", initial_sidebar_state="expanded")
st.markdown("<h1 style='color:white; background-color: green;text-align: center;'>Welcome to Hand Gesture to Speech Conversion!...</h1>", unsafe_allow_html=True)
#st.markdown("<p style='color: black;text-align: center;font-size: 20px;'>This web application allows you to communicate using Indian Sign Language, which is used by millions of people in India who are deaf or hard-of-hearing. By using this application, you can type messages that will be displayed in sign language or use your hands to create gestures that will be translated into text messages. This application is meant to bridge the communication gap between those who know sign language and those who do not, making communication easier and more accessible for everyone.</p>", unsafe_allow_html=True)
st.write("The ability to communicate is essential for everyone, but for people with disabilities, communication can be a "
         "challenge. In India, the 2011 Census recorded a total of 26.8 million people with disabilities, of which 7% had "
         "speech impairments. Technologies that enable communication, such as Indian Sign Language recognition, can help "
         "bridge the communication gap and make communication more accessible for everyone.")
# Add a streamlit video display for the webcam feed
video_display = st.empty()

# Load the hand detector and classification models
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300
labels = ["A", "B", "C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"," "]
labels2 = ["Where is this address?",
           "Where is the bus stop?",
           "Can you recommend a good coffee shop?",
           "Do you know where the nearest drugstore is?",
           "Excuse me, could you tell me where the elevator is?",
           "Can you help me find a good fish market?",
           "Could you give me directions to the grocery store?",
           "Can you recommend a good hotel nearby?",
           "Is there a good Italian restaurant in the area?",
           "Do you know where the nearest Japanese restaurant is?",
           "Can you tell me where the nearest kindergarten is?",
           "Where is the nearest library?",
           "Can you recommend a good Mexican restaurant?",
           "Do you know where the nearest newsstand is?",
           "Where is the nearest office supply store?",
           "Do you know where the nearest post office is?",
           "Are there any good quality restaurants in the area?",
           "Where is the nearest railway station?",
           "Can you recommend a good seafood restaurant?",
           "Where is the nearest taxi stand?",
           "Do you know where the nearest university is?",
           "Can you tell me where the nearest veterinarian is?",
           "Where is the nearest wine shop?",
           "I'm not sure what to ask for X, sorry!",
           "Do you know where the nearest yoga studio is?",
           "Can you recommend a good zoo in the area?",
           " "]

sahil=1
old_index=99
start_button = st.button("Start")
stop_button = st.button("Stop")

def speak(text):
    global flag
    if engine._inLoop:
        engine.endLoop()
    engine.say(text)
    engine.startLoop(False)
    while engine.iterate():
        pass
    flag = False

while True:
    if start_button:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand1, hand2 = hands[0], hands[1] if len(hands) > 1 else None

            # Check the position of the hands detected (always detect left hand first)
            if hand2 and hand2["bbox"][0] < hand1["bbox"][0]:
                hand1, hand2 = hand2, hand1

            x1, y1, w1, h1 = hand1['bbox']
            imgWhite = np.ones((imgSize * 2, imgSize * 2, 3), np.uint8) * 255

            # Add boundary checks to ensure hand1 is within the frame
            if x1 - offset >= 0 and y1 - offset >= 0 and x1 + w1 + offset < img.shape[1] and y1 + h1 + offset < img.shape[0]:
                imgCrop1 = img[y1 - offset:y1 + h1 + offset, x1 - offset:x1 + w1 + offset]

                imgCropShape = imgCrop1.shape

                aspectRatio1 = h1 / w1

                if aspectRatio1 > 1:
                    k = imgSize / h1
                    wCal1 = math.ceil(k * w1)
                    imgResize1 = cv2.resize(imgCrop1, (wCal1, imgSize))
                    imgResizeShape = imgResize1.shape
                    wGap1 = math.ceil((imgSize - wCal1) / 2)
                    imgWhite[:imgSize, wGap1:wCal1 + wGap1] = imgResize1

                else:
                    k = imgSize / w1
                    hCal1 = math.ceil(k * h1)
                    imgResize1 = cv2.resize(imgCrop1, (imgSize, hCal1))
                    imgResizeShape = imgResize1.shape
                    hGap1 = math.ceil((imgSize - hCal1) / 2)
                    imgWhite[hGap1:hCal1 + hGap1, :imgSize] = imgResize1

            if hand2:
                x2, y2, w2, h2 = hand2['bbox']
                # Add boundary checks to ensure hand2 is within the frame
                if x2 - offset >= 0 and y2 - offset >= 0 and x2 + w2 + offset < img.shape[1] and y2 + h2 + offset < img.shape[0]:
                    imgCrop2 = img[y2 - offset:y2 + h2 + offset, x2 - offset:x2 + w2 + offset]

                    imgCropShape = imgCrop2.shape

                    aspectRatio2 = h2 / w2

                    if aspectRatio2 > 1:
                        k = imgSize / h2
                        wCal2 = math.ceil(k * w2)
                        imgResize2 = cv2.resize(imgCrop2, (wCal2, imgSize))
                        imgResizeShape = imgResize2.shape
                        wGap2 = math.ceil((imgSize - wCal2) / 2)
                        imgWhite[imgSize:, imgSize + wGap2:wCal2 + imgSize + wGap2] = imgResize2

                    else:
                        k = imgSize / w2
                        hCal2 = math.ceil(k * h2)
                        imgResize2 = cv2.resize(imgCrop2, (imgSize, hCal2))
                        imgResizeShape = imgResize2.shape
                        hGap = math.ceil((imgSize - hCal2) / 2)
                        imgWhite[hGap:hCal2 + hGap, imgSize:] = imgResize2

            img1 = cv2.flip(img, 1)
            prediction, index = classifier.getPrediction(imgWhite)
            # video_display.image(imgWhite, channels="BGR") #.......use to check imgWhite

            if prediction == "L":
                pyautogui.press('space')
            elif prediction == "V":
                pyautogui.press('v')

            cv2.rectangle(imgOutput, (0, 0),
                          (60, 60), (255, 255, 255), cv2.FILLED)
            if index >= 0 and index < len(labels):
                cv2.putText(imgOutput, labels[index], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            if sahil == 1 or old_index != index:
                old_index = index
                speak(labels2[index])
                sahil = 0

    if stop_button:
        # Break out of the loop
        break

    # Display the video in the stframe
    video_display.image(imgOutput, channels="BGR")
