import cv2
from ultralytics import YOLO
import torch
import numpy as np

background = "windows.jpg"
people = "colorful"
saturation_scale = 2.0
width, height = 1920, 1080

if "." in background:
    background_frame = cv2.imread(f"data/{background}")
    background_frame = cv2.resize(background_frame, (width, height))
    if len(background_frame.shape) == 2:
        resized_image = cv2.cvtColor(background_frame, cv2.COLOR_GRAY2BGR)

# Load the YOLOv8 model
model = YOLO('yolov8x-seg')

# Start streaming from camera 0
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        if results[0].masks is not None:
            class_labels = results[0].boxes.cls
            masks = results[0].masks.data
            masks = masks[class_labels == 0]
            if len(masks) > 0:
                res_tensor = sum(masks.data)  # + results[0].masks.faces[1] + results[0].masks.faces[2])
                res_tensor = torch.where(res_tensor != 0, torch.ones_like(res_tensor), res_tensor)

                # Mask of segmented results
                mask = res_tensor.cpu().numpy()
                mask = np.stack([mask, mask, mask], axis=2)

                if mask.shape != (height, width, 3):
                    mask = cv2.resize(mask, (width, height))

                # Mask padding
                shift_x = 0
                shift_y = 0
                bordered_mask = np.zeros((mask.shape[0] + shift_y, mask.shape[1] + shift_x, 3), dtype=np.uint8)
                bordered_mask[shift_y:, shift_x:] = mask
                mask = bordered_mask[:mask.shape[0], :mask.shape[1]]

                if background == "gray":
                    # Converting frame to gray scale and excluding detected objects
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
                    frame = np.where(mask == 0, gray_frame, frame)
                elif background == "black":
                    black_frame = np.zeros_like(frame)
                    frame = np.where(mask == 0, black_frame, frame)
                elif background != "":
                    frame = np.where(mask == 0, background_frame, frame)

                if people == "colorful":
                    # More color saturation on detected objects
                    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    hsv_frame[:, :, 1] = cv2.multiply(hsv_frame[:, :, 1], saturation_scale)
                    saturated_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
                    frame = np.where(mask == 1, saturated_frame, frame)

        # Display frame
        cv2.imshow("YOLOv8", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
