import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime, timedelta
import csv

path = 'student_images'
images = []
classNames = []
mylist = os.listdir(path)
for cl in mylist:
    # Skip any files that are not image files
    if not cl.endswith(('.jpg', '.jpeg', '.png')):
        continue

    curImg = cv2.imread(f'{path}/{cl}')
    if curImg is None:
        print(f"Error loading image: {cl}")
        continue

    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(img)[0]
        encodeList.append(encoded_face)
    return encodeList

encoded_face_train = findEncodings(images)

# Dictionary to store the last attendance time for each student
last_attendance_time = {}



#csv file creation and manipulation

def markAttendance(name):
    today_date = datetime.now().strftime('%d-%B-%Y')
    with open('Attendance.csv', 'r') as f:
        reader = csv.reader(f)
        # Check if the person has already been marked present today
        for row in reader:
            if len(row) >= 3 and row[0] == name and row[2] == today_date:
                print(f'{name} has already been marked present today.')
                return

    # If person has not been marked present today, add a new entry
    with open('Attendance.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        now = datetime.now()
        current_time = now.strftime('%I:%M:%S %p')
        writer.writerow([name, current_time, today_date])
        print(f'{name} marked present at {current_time} on {today_date}.')

# take pictures from webcam 
cap  = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS)
    encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)
    for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
        matches = face_recognition.compare_faces(encoded_face_train, encode_face)
        faceDist = face_recognition.face_distance(encoded_face_train, encode_face)
        matchIndex = np.argmin(faceDist)
        print(matchIndex)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper().lower()
            y1, x2, y2, x1 = faceloc


            # since we scaled down by 4 times
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
    cv2.imshow('webcam', img)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
