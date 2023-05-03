import cv2
import easyocr


def main():
    # Create an OCR reader instance
    reader = easyocr.Reader(['en'], gpu=True)

    # Use the first available camera
    cap = cv2.VideoCapture(0)
    width, height = 1920, 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while True:
        # Read the image from the camera
        ret, frame = cap.read()

        # Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Read text from the image
        results = reader.readtext(gray)

        # Display OCR results on the image
        for result in results:
            x1, y1 = int(result[0][0][0]), int(result[0][0][1])
            x2, y2 = int(result[0][2][0]), int(result[0][2][1])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, result[1], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the image with the camera
        cv2.imshow('Real-time OCR', frame)

        # Exit if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera resources and close the window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
