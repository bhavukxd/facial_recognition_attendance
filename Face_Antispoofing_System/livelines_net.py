import cv2
import os
import numpy as np
from tensorflow.keras.models import model_from_json
import face_recognition
import csv
from datetime import datetime, timedelta

face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

with open('antispoofing_models/antispoofing_model_mobilenet.json', 'r') as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights('antispoofing_models/antispoofing_model_86-1.000000.weights.h5')

known_face_encodings = []
known_face_names = []
database_path = "C:/Users/BHAVUK/OneDrive/Desktop/antispoofing/Face_Antispoofing_System/databas"
for person_name in os.listdir(database_path):
    person_folder = os.path.join(database_path, person_name)
    if os.path.isdir(person_folder):
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(person_name)

attendance_folder = "attendance_records"
os.makedirs(attendance_folder, exist_ok=True)

last_attendance_time = {}
attendance_interval = timedelta(minutes=5)

def log_attendance(name):
    date_today = datetime.now().strftime("%Y-%m-%d")
    file_path = os.path.join(attendance_folder, f"{date_today}.csv")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Name", "Timestamp"])
        writer.writerow([name, timestamp])

video = cv2.VideoCapture(0)

while True:
    try:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y-5:y+h+5, x-5:x+w+5]
            resized_face = cv2.resize(face, (160, 160))
            resized_face = resized_face.astype("float") / 255.0
            resized_face = np.expand_dims(resized_face, axis=0)
            preds = model.predict(resized_face)[0]
            if preds > 0.5:
                label = 'spoof'
                color = (0, 0, 255)
            else:
                label = 'real'
                color = (0, 255, 0)
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(face_rgb)
                name = "Unknown"
                if face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[0])
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        current_time = datetime.now()
                        if name not in last_attendance_time or (current_time - last_attendance_time[name]) > attendance_interval:
                            log_attendance(name)
                            last_attendance_time[name] = current_time
                label = name if name != "Unknown" else "spoof"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"Error: {e}")

video.release()
cv2.destroyAllWindows()