import cv2

def image():
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
    img = cv2.imread('test1.jpg')

# Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the output
    cv2.imshow('img', img)
    cv2.waitKey()

def Video():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0) # capturing the video from camera
    while True:
    # Read the frame
     _, img = cap.read()

    # Convert to grayscale
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
     faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
     for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display
     cv2.imshow('img', img)

    # Stop if escape key is pressed
     k = cv2.waitKey(0) & 0xff
     if k==27:
        break
        
# Release the VideoCapture object
    cap.release()

choice=input("Enter the 1 For image and 2 for Video")
if choice == '1':
    image()
elif choice == '2':
    Video()
    