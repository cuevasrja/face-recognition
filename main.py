import cv2

#Let's load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#here we will initialize video capture (0 for built-in webcam)
cap = cv2.VideoCapture(0)

while True:
    #this will read frame-by-frame from the webcam
    ret, frame = cap.read()
    
    #Convert frame to grayscale (Haar Cascade works better with grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    #this will draw rectangles around detected faces as shown
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    #display the resulting frame with detected faces
    cv2.imshow('Face Detection', frame)
    
    #break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release the video capture and close windows
cap.release()
cv2.destroyAllWindows()