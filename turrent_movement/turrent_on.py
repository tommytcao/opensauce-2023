import cv2
import numpy as np
import serial
import time
import threading

angle_lock = threading.Lock()
x_angle_global, y_angle_global = 90, 90
laser_on = False  # Add a variable to keep track of the laser state
import json
def load_settings():
    with open("settings.json") as file:
        settings = json.load(file)
    return settings

settings = load_settings()


# Connect to serial port
ser = serial.Serial(settings['hand_usb'], 9600, write_timeout=1)
stop_flag = threading.Event()


def move_servos():

    while True:
        with angle_lock:

            for servo_id in servo_angles_global.keys():
                command = f"{servo_id}{int(servo_angles_global[servo_id]['x_angle_global'])},{int(servo_angles_global[servo_id]['y_angle_global'])}\n"

                ser.write(command.encode())
            time.sleep(0.1)
        if stop_flag.is_set():
            break


servo_angles_global = {
    1: {"x_angle_global": 90, "y_angle_global": 90},
    2: {"x_angle_global": 90, "y_angle_global": 90},
    3: {"x_angle_global": 90, "y_angle_global": 90},
    4: {"x_angle_global": 90, "y_angle_global": 90},
    5: {"x_angle_global": 90, "y_angle_global": 90},
}
# Load the precomputed servo angles from the .npy file
servo_angles_map = {
    1: np.load("servo_angles1.npy"),
    2: np.load("servo_angles2.npy"),
    3: np.load("servo_angles3.npy"),
    4: np.load("servo_angles4.npy"),
    5: np.load("servo_angles5.npy"),
}


servo_thread = threading.Thread(target=move_servos)
servo_thread.daemon = True
servo_thread.start()


# Define the mouse callback function to handle mouse events
def mouse_callback(event, x, y, flags, param):
    global x_angle_global, y_angle_global
    if (
        not face_detection_mode
    ):  # Only allow mouse movement when not in face detection mode
        if event == cv2.EVENT_MOUSEMOVE:
            with angle_lock:

                for servo_id in servo_angles_global.keys():

                    (
                        servo_angles_global[servo_id]["y_angle_global"],
                        servo_angles_global[servo_id]["x_angle_global"],
                    ) = servo_angles_map[servo_id][y, x]


# Initialize the camera capture
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Set the mouse callback function for the window
cv2.namedWindow("Camera Feed")
cv2.setMouseCallback("Camera Feed", mouse_callback)
face_detection_mode = False
width = 640
height = 480
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
text_color = (0, 0, 255)  # Red color

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (width, height))

    if face_detection_mode:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            center_x, center_y = x + w // 2, y + h // 2
            with angle_lock:
                for servo_id in servo_angles_global.keys():

                    (
                        servo_angles_global[servo_id]["y_angle_global"],
                        servo_angles_global[servo_id]["x_angle_global"],
                    ) = servo_angles_map[servo_id][center_y, center_x]
            
    # Resize the frame to the desired width and height


    # Display the resulting frame
    
    mode_display = "Face Detection Mode" if face_detection_mode else "Mouse Control Mode"
    cv2.putText(frame, mode_display, (10, 30), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.imshow("Camera Feed", frame)
    # Handle key press events
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        stop_flag.set()
        servo_thread.join()
        break
    elif key == ord("l"):  # Toggle the laser when the 'l' key is pressed
        laser_on = not laser_on
        ser.write(
            ("L%d\n" % int(laser_on)).encode()
        )  # Send the laser command to the Arduino
    elif key == ord("f"):  # Toggle the face detection mode when the 'f' key is pressed
        face_detection_mode = not face_detection_mode
    
# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
