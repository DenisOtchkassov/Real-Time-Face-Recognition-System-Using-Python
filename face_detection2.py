import cv2
from deepface import DeepFace
import numpy as np

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Preload images of known individuals
database = {
    "Denis": r"C:\Users\denis\Desktop\Faces\DenisImage.jpg",
}

# Pre-compute embeddings for known individuals
known_faces = {}
for name, img_path in database.items():
    # Generate embedding for each person
    embedding = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
    known_faces[name] = np.array(embedding)

# Function to calculate cosine similarity between two vectors
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face region of interest (ROI)
        face_img = frame[y:y+h, x:x+w]
        
        try:
            # Get the embedding for the detected face
            face_embedding = DeepFace.represent(img_path=face_img, model_name="Facenet", enforce_detection=False)[0]["embedding"]
            face_embedding = np.array(face_embedding)
            
            # Initialize variables to store the best match
            name = "Unknown"
            highest_similarity = -1

            # Compare the detected face with known faces
            for person, embedding in known_faces.items():
                similarity = cosine_similarity(face_embedding, embedding)
                if similarity > highest_similarity and similarity > 0.8:  # Threshold for recognition
                    highest_similarity = similarity
                    name = person

        except Exception as e:
            # Handle exceptions and set the name as error
            name = f"Error: {str(e)}"
            print("Exception:", e)

        # Draw a rectangle around the detected face and display the name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the frame with face recognition results
    cv2.imshow('Face Recognition', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
