import cv2
import numpy as np
import torch
from PIL import Image
import mediapipe as mp
import pyttsx3
import time
from utils import getTransform, load_model, detect_and_crop_hand

class SignLanguageRecognizer:
    def __init__(self, model_path, label_map):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = SignLanguageCNN(num_classes=28)  # Number of classes
        # self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        # self.model.to(self.device)
        # self.model.eval()

        self.model = load_model(model_path, self.device)
        
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        self.engine = pyttsx3.init()
        self.sentence_buffer = ''
        self.last_time_hand_seen = time.time()
        self.last_predicted_label = None
        self.last_prediction_time = 0
        self.prediction_duration_threshold = 2

        self.label_map = label_map

        self.transform = getTransform(size=128)


    def speak_text(self, text):
        """Convert text to speech."""
        self.engine.say(text)
        self.engine.runAndWait()


    # def thresholding(self, pil_image):
    #     image = np.array(pil_image)
    #     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #     binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                    cv2.THRESH_BINARY_INV, 11, 2)
    #     kernel = np.ones((3, 3), np.uint8)
    #     binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    #     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     for cnt in contours:
    #         if cv2.contourArea(cnt) < 200:
    #             cv2.drawContours(binary, [cnt], 0, 0, -1)
        
    #     binary = cv2.bitwise_not(binary)
    #     cv2.imshow("Transformed Binary", binary)
    #     return Image.fromarray(binary)


    # def detect_and_crop_hand(self, frame):
    #     """Detect hands using MediaPipe and crop the image around the detected hand."""
    #     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     results = self.mp_hands.process(image_rgb)
    #     if results.multi_hand_landmarks:
    #         for hand_landmarks in results.multi_hand_landmarks:
    #             # Assuming only one hand per image for simplicity, more specifically right hand.
    #             x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
    #             x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
    #             y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
    #             y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
    #             x_min, x_max = max(0, x_min - 50), min(frame.shape[1], x_max + 50)
    #             y_min, y_max = max(0, y_min - 50), min(frame.shape[0], y_max + 50)
    #             return frame[int(y_min):int(y_max), int(x_min):int(x_max)], True
    #     return frame, False


    def predict_sign_language(self, frame):
        pil_image = Image.fromarray(frame)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            _, predicted = torch.max(output, 1)
            predicted_label = self.label_map[predicted.item()]
        return predicted_label


    def update_sentence_buffer(self, predicted_label):
        current_time = time.time()
        if predicted_label != self.last_predicted_label:
            self.last_predicted_label = predicted_label
            self.last_prediction_time = current_time
        elif current_time - self.last_prediction_time >= self.prediction_duration_threshold:
            if predicted_label == 'space':
                self.sentence_buffer += ' '
            elif predicted_label == 'delete':
                self.sentence_buffer = self.sentence_buffer[:-1]
            else:
                self.sentence_buffer += predicted_label
            
            self.last_prediction_time = current_time

        if current_time - self.last_time_hand_seen > 5.0:
            if self.sentence_buffer.strip():
                print("Final sentence:", self.sentence_buffer)
                self.speak_text(self.sentence_buffer)
                self.sentence_buffer = ''
        self.last_time_hand_seen = current_time


    def run(self):
        cap = cv2.VideoCapture(0)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                cropped_frame, found = detect_and_crop_hand(frame, self.mp_hands)
                if found:
                    label = self.predict_sign_language(cropped_frame)
                    self.update_sentence_buffer(label)
                    info_text = f'Label: {label}, Sentence: {self.sentence_buffer}'
                    cv2.putText(frame, info_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                                cv2.LINE_AA)
                    cv2.imshow('Cropped Hand', cropped_frame)
                else:
                    cv2.putText(frame, "No hand detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)

                cv2.imshow('Sign Language Recognition', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.mp_hands.close()


if __name__ == "__main__":
    label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'delete', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'space', 21: 'T', 22: 'U', 23: 'V', 24: 'W', 25: 'X', 26: 'Y', 27: 'Z'}
    recognizer = SignLanguageRecognizer('sign_language_cnn_space_delete_good.pth', label_map)
    recognizer.run()