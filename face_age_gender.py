import cv2
import numpy as np

# Define model paths
face_pbtxt = "models/opencv_face_detector.pbtxt"
face_pb = "models/opencv_face_detector_uint8.pb"
age_prototxt = "models/age_deploy.prototxt"
age_model = "models/age_net.caffemodel"
gender_prototxt = "models/gender_deploy.prototxt"
gender_model = "models/gender_net.caffemodel"
MODEL_MEAN_VALUES = [104, 117, 123]
# Load models
face_net = cv2.dnn.readNet(face_pb, face_pbtxt)
age_net = cv2.dnn.readNet(age_model, age_prototxt)
gender_net = cv2.dnn.readNet(gender_model, gender_prototxt)
# Classifications
age_classifications = ['(0-10)', '(11-20)', '(21-30)', '(31-40)', '(41-50)', '(51-60)', '(61-70)', '(71-80)', '(81-90)', '(91-100)']
gender_classifications = ['female', 'male']
# Function for face detection and predictions
def detect_and_predict(frame, use_camera=True):
    img_h, img_w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False, crop=False)
    # Face detection
    face_net.setInput(blob)
    detections = face_net.forward()
    face_bounds = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:  # Confidence threshold
            x1 = int(detections[0, 0, i, 3] * img_w)
            y1 = int(detections[0, 0, i, 4] * img_h)
            x2 = int(detections[0, 0, i, 5] * img_w)
            y2 = int(detections[0, 0, i, 6] * img_h)
            # Draw rectangle around face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            face_bounds.append([x1, y1, x2, y2])
    if not face_bounds:
        print("No faces detected.")
        return frame
    # Process each detected face
    for face_bound in face_bounds:
        try:
            x1, y1, x2, y2 = face_bound
            face_roi = frame[max(0, y1 - 15):min(y2 + 15, img_h - 1), max(0, x1 - 15):min(x2 + 15, img_w - 1)]
            # Create blob for age and gender prediction
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=True, crop=False)
            # Gender prediction
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_classifications[gender_preds[0].argmax()]
            gender_confidence = gender_preds[0].max()
            # Age prediction
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_classifications[age_preds[0].argmax()]
            age_confidence = age_preds[0].max()
            # Display results
            label = f"{gender}, {age}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Gender Accuracy: {gender_confidence:.2f}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(frame, f"Age Accuracy: {age_confidence:.2f}", (x1, y2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        except Exception as e:
            print(f"Error processing face: {e}")
            continue
    return frame
# Main function to run webcam or process an uploaded image
def main():
    choice = input("Choose input method: (1) Webcam (2) Image File: ")  
    if choice == '1':  # Webcam
        video = cv2.VideoCapture(0)
        while cv2.waitKey(1) < 0:
            hasFrame, frame = video.read()
            if not hasFrame:
                cv2.waitKey()
            resultImg = detect_and_predict(frame, use_camera=True)
            cv2.imshow("Age and Gender Detection", resultImg)
            if cv2.waitKey(33) & 0xFF == ord('q'):  # Press 'q' to exit
                break
        video.release()
        cv2.destroyAllWindows()
    elif choice == '2':  # Image file
        image_path = input("Enter image file path: ")
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Unable to load image.")
            return
        image = cv2.resize(image, (720, 640))
        resultImg = detect_and_predict(image, use_camera=False)
        cv2.imshow("Age and Gender Detection", resultImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Invalid choice. Exiting...")
# Run the program
if __name__ == "__main__":
    main()
