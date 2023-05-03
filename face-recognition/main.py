import os
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

device = torch.device("cuda:0")

# Initialize MTCNN for face detection
mtcnn = MTCNN(device=device)

# Initialize FaceNet for feature vector generation
facenet = InceptionResnetV1(pretrained='vggface2').eval()
facenet = facenet.to(device)

# Initialize the camera (default camera: 0)
cap = cv2.VideoCapture(0)

# Load the Haar cascade model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


face_embeddings = {}
filenames = os.listdir("faces")
for filename in filenames:
    # Load images for comparison
    img = cv2.imread(f"faces/{filename}")
    # Detect and crop faces
    face_img = mtcnn(img)
    # Move the image to GPU
    face_img = face_img.to(device)
    # Extract feature vectors
    face_embedding = facenet(face_img.unsqueeze(0))
    face_embeddings[filename[:-4]] = face_embedding


while True:
    # Capture a single frame from the camera
    ret, frame = cap.read()

    # Process the frame (e.g., convert to grayscale)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    # Compare the detected face with the database of faces
    for face in faces:
        x, y, w, h = face
        face_roi = frame[y:y + h, x:x + w]
        face_detected_img = mtcnn(face_roi)
        # Set default text and frame color
        text = "Unknown"
        color = (0, 0, 255)
        # If both faces are detected
        if face_detected_img is not None:
            # Move the image to GPU
            face_detected_img = face_detected_img.to(device)
            # Extract feature vector
            face_detected_embedding = facenet(face_detected_img.unsqueeze(0))

            for k in face_embeddings.keys():
                # Calculate the distance between feature vectors
                distance = (face_embeddings[k] - face_detected_embedding).norm().item()

                # Compare the distance with the threshold
                threshold = 0.9  # The threshold value can be adjusted
                if distance < threshold:
                    text = k
                    color = (255, 0, 0)
                    break

        # Draw rectangles around detected faces
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        # Add description (text) to the rectangle

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        cv2.putText(frame, text, (x, y - 10), font, font_scale, color, font_thickness)

    # Display the processed frame
    cv2.imshow('Live Video', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera resources and close windows
cap.release()
cv2.destroyAllWindows()
