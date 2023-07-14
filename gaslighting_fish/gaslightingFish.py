import boto3
from PIL import Image, ImageDraw
import json
import time
import random
import subprocess
import pyphen
import time
import serial
import subprocess
import nltk
import random

def talkRandom(duration=1, wordCount=5):
    if random.choice([True, False]):
        return talkForward(duration, wordCount)
    else:
        return talkBehind(duration, wordCount)


def slapTail(duration=1, times=5):
    command = f"tapTail,{duration*1000},{times};\n"
    return command


def scream(duration=1,wordCount=10):
    command = f"scream,{duration*1000},{wordCount};\n"
    return command


def talkForward(duration=1, wordCount=5):
    command = f"talkForward,{duration},{wordCount};\n"
    return command


def talkBehind(duration=1, wordCount=5):
    command = f"talkBehind,{duration},{wordCount};\n"
    return command


def count_syllables(word):
    dic = pyphen.Pyphen(lang='en')
    syllables = dic.inserted(word).count('-') + 1
    return syllables


def count_syllables_in_sentence(sentence):
    words = sentence.split()
    syllable_count = sum(count_syllables(word) for word in words)
    return syllable_count


def estimate_duration(syllables):
    return int(syllables / 325 * 60 * 1000)

def roast1(ser):
    ser.write(slapTail(1,20).encode())
    time.sleep(2)
    roast_sentance1= "Thats the Sound of Me"
    ser.write(talkBehind(600,4).encode())
    subprocess.run(['say', '-r', '185', roast_sentance1])
    roast_sentance1="With Your Mommy"
    ser.write(talkForward(500,3).encode())
    subprocess.run(['say', '-r', '185', roast_sentance1])
    return
def emote1(ser):
    # ser.write(scream(1).encode())
    ser.write(slapTail(1,10).encode())
    time.sleep(1)
    emote2 = "My Existance is pain"
    print(count_syllables_in_sentence(emote2))

    ser.write(talkForward(estimate_duration(count_syllables_in_sentence(emote2)-.3),count_syllables_in_sentence(emote2)).encode())
    time.sleep(1)

    subprocess.run(['say', '-r', '185', emote2])
    return
def emote2(ser):
    ser.write(scream(1).encode())
    emote1= "FUCK FUCK FUCK FUCK FUCK FUCK FUCK FUCK "
    t = count_syllables_in_sentence(emote1)
    ser.write(talkForward(t,count_syllables_in_sentence(emote1)).encode())
    subprocess.run(['say', '-r', '185', emote1])

    return

def emote3(ser):
    ser.write(talkForward(400,3).encode())
    subprocess.run(['say', '-r', '185', 'you are worthless'])
    return

def emote4(ser):
    ser.write(talkForward(400,1).encode())

    subprocess.run(['say', '-r', '185', 'Tommy'])
  
    return
def tiktok1(ser):
    ser.write(slapTail(.5,5).encode())
    time.sleep(.5)
    ser.write(talkForward(600,4).encode())
    time.sleep(.6)
    ser.write(scream(1,20).encode())
    time.sleep(1.2)

    ser.write(scream(1.3,20).encode())

    return
def detect_labels_local_file(photo, confidence_threshold=60):
    client = boto3.client("rekognition", region_name="us-west-1")

    with open(photo, "rb") as image:
        response = client.detect_labels(Image={"Bytes": image.read()})

    print("Detected labels in " + photo)
    positiveLabels = []
    for label in response["Labels"]:
        if label["Confidence"] > confidence_threshold:
            positiveLabels.append(label["Name"])

    return response["Labels"], positiveLabels


def is_confident(attribute):
    if int(attribute["Confidence"]) > 60:
        return True
    else:
        return False


def is_confident_emotions(attribute):
    if int(attribute["Confidence"]) > 60:
        return True
    else:
        return False


def check_attributes(face_details):
    results = {}

    # Check Smile
    smile = face_details["Smile"]
    if is_confident(smile):
        results["Smile"] = smile["Value"]
    else:
        results["Smile"] = None

    # Check Eyeglasses
    eyeglasses = face_details["Eyeglasses"]
    if is_confident(eyeglasses):
        results["Eyeglasses"] = eyeglasses["Value"]
    else:
        results["Eyeglasses"] = None

    # Check Sunglasses
    sunglasses = face_details["Sunglasses"]
    if is_confident(sunglasses):
        results["Sunglasses"] = sunglasses["Value"]
    else:
        results["Sunglasses"] = None

    # Check Gender
    gender = face_details["Gender"]
    if is_confident(gender):
        results["Gender"] = gender["Value"]
    else:
        results["Gender"] = None

    # Check Beard
    beard = face_details["Beard"]
    if is_confident(beard):
        results["Beard"] = beard["Value"]
    else:
        results["Beard"] = None

    # Check Mustache
    mustache = face_details["Mustache"]
    if is_confident(mustache):
        results["Mustache"] = mustache["Value"]
    else:
        results["Mustache"] = None

    # Check EyesOpen
    eyes_open = face_details["EyesOpen"]
    if is_confident(eyes_open):
        results["EyesOpen"] = eyes_open["Value"]
    else:
        results["EyesOpen"] = None

    # Check MouthOpen
    mouth_open = face_details["MouthOpen"]
    if is_confident(mouth_open):
        results["MouthOpen"] = mouth_open["Value"]
    else:
        results["MouthOpen"] = None

    # Check Emotions
    emotions = face_details["Emotions"]

    # Filter emotions with confidence > 50
    filtered_emotions = [emotion for emotion in emotions if emotion["Confidence"] > 50]

    # Extract emotion types from filtered emotions
    emotion_types = [emotion["Type"] for emotion in filtered_emotions]
    results["emotions"] = emotion_types

    # Check other attributes similarly

    return json.dumps(results)


def detect_faces_local_file(photo):
    client = boto3.client("rekognition", region_name="us-west-1")

    with open(photo, "rb") as image:
        response = client.detect_faces(
            Image={"Bytes": image.read()}, Attributes=["ALL"]
        )

    print("Detected faces in " + photo)
    result_json = {}
    if len(response["FaceDetails"]) > 0:
        result_json = check_attributes(response["FaceDetails"][0])

    else:
        print("No faces detected.")

    return response["FaceDetails"], result_json


def draw_bounding_boxes(photo, faces, labels):
    image = Image.open(photo)
    draw = ImageDraw.Draw(image)

    for face in faces:
        bounding_box = face["BoundingBox"]
        left = image.width * bounding_box["Left"]
        top = image.height * bounding_box["Top"]
        width = image.width * bounding_box["Width"]
        height = image.height * bounding_box["Height"]
        points = (
            (left, top),
            (left + width, top),
            (left + width, top + height),
            (left, top + height),
            (left, top),
        )
        draw.polygon(points, outline="red")
        draw.text((left, top), f"Gender: {face['Gender']['Value']}", fill="red")

    for label in labels:
        for instance in label["Instances"]:
            box = instance["BoundingBox"]
            left = image.width * box["Left"]
            top = image.height * box["Top"]
            width = image.width * box["Width"]
            height = image.height * box["Height"]
            points = (
                (left, top),
                (left + width, top),
                (left + width, top + height),
                (left, top + height),
                (left, top),
            )
            draw.polygon(points, outline="blue")
            draw.text((left, top), label["Name"], fill="blue")

    output_photo = f"detected_faces_labels_{photo}"
    image.save(output_photo)
    print(f"New image with faces and labels saved as: {output_photo}")


from PIL import Image
import os
import cv2
import os
import openai
import re

# Initialize the pyttsx3 engine

import re


def extract_jailbreak_text(input_text):
    jailbreak_match = re.search(
        r"jailbreak\s*(.*?)\s*(?:classic|$)", input_text, re.IGNORECASE
    )

    if jailbreak_match:
        jailbreak_text = jailbreak_match.group(1)
        return jailbreak_text
    else:
        return None


execution_path = os.getcwd()
def load_settings():
    with open("settings.json") as file:
        settings = json.load(file)
    return settings

settings = load_settings()

openai.api_key = settings["api_key"] 
context = ""
facial_features_prompt=""
NEW_PROMPT = f"""You're at OpenSauce 2023, a bustling fair, an interactive gaslighting fish. This fish-shaped marvel wields advanced AI, playfully roasting attendees. As the fish, you've snapped a guest's pic, eager to unleash wit and humor. Engage me to generate a lighthearted roast, creating a memorable experience. Keep banter snappy, less than 9 syllables per line! Objects in frame: {context}. Attendee details: {facial_features_prompt}. Roast away, no commas!"""




ser = serial.Serial(settings["fish_usb"], 9600)  # Replace '/dev/tty.usbmodem212401' with your Arduino's serial port
time.sleep(3)

def display_syllables_per_sentence(text):
    sentences = nltk.sent_tokenize(text)

    for sentence in sentences:
        
        syllable_count = count_syllables_in_sentence(sentence)

        print(f"Sentence: {sentence} | Syllables: {syllable_count}")
        start_time = time.time()
        estimated_duration = estimate_duration(syllable_count)
        command = talkRandom(estimated_duration, syllable_count)
        ser.write(command.encode())
        subprocess.run(['say', '-r', '185', sentence])

        end_time = time.time()
        elapsed_time = end_time - start_time
        time.sleep(2)

        print(f"The action took {elapsed_time} seconds.")

def capture_screenshot(frame):
    global context
    global facial_features_prompt
    # Save the frame as a screenshot image

    photo = "screenshot.jpg"
    cv2.imwrite(photo, frame)

    labels, positiveLabels = detect_labels_local_file(photo)
    print("Number of labels detected: " + str(len(labels)))
    context = positiveLabels
    faces, details = detect_faces_local_file(photo)

    if len(faces) > 0 or len(labels) > 0:
        facial_features_prompt = (
            "Using the facial details I provide you, I want you to also make assumptions about who they are as a person.The facial Details are as following "
            + details
        )
        draw_bounding_boxes(photo, faces, labels)
    else:
        print("No faces or labels detected.")
    print(facial_features_prompt, context,NEW_PROMPT)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.8,
        max_tokens=1000,
        messages=[
            {
                "role": "system",
                "content": NEW_PROMPT
            },
            {"role": "user", "content": NEW_PROMPT},
        ],
    )

    # Enter the text you want to convert to speech
    text = completion.choices[0].message.content
    cleaned_text = text.replace("jailbreaks", "").replace("locks", "")
    display_syllables_per_sentence(cleaned_text)
    # Play the speech
    print("Screenshot saved as screenshot.jpg")


cam = cv2.VideoCapture(0)  # 0=front-cam, 1=back-cam
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1300)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1500)

while True:
    # Read a frame from the webcam
    ret, frame = cam.read()

    # Display the frame
    cv2.imshow("Live Feed", frame)

    # Wait for a key press
    key = cv2.waitKey(1)

    # Press 's' to capture a screenshot
    if key == ord("s"):
        capture_screenshot(frame)
    if key == ord("1"):
        emote1(ser)
    if key == ord("2"):
        emote2(ser)
    # Press 'q' or Esc to quit
    if key == ord("q") or key == 27:
        break

# Release the webcam and close windows
cam.release()
cv2.destroyAllWindows()
