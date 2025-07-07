import cv2


def take_image():
    print("Press 's' to save the image or 'q' to quit.")
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:
        try:
            check, frame = webcam.read()
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('s'):
                cv2.destroyAllWindows()
                return frame
            elif key == ord('q'):
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                return None
        except KeyboardInterrupt:
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            return None


if __name__ == "__main__":
    print("This script captures an image from the webcam.")
    img = take_image()
    if img is not None:
        print("Image captured successfully.")
    else:
        print("No image captured.")
