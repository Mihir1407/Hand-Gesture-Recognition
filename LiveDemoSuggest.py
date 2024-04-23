import cv2
import numpy as np
import torch
from PIL import Image
import mediapipe as mp
import pyttsx3
import time
import nltk
from nltk.corpus import words
from utils import getTransform, load_model, detect_and_crop_hand

class SignLanguageRecognizerWithSuggestions:
    def __init__(self, model_path, label_map):
        self.num_classes = 28
        # self.model = SignLanguageCNN(num_classes=self.num_classes)
        # self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(model_path, self.device)
        
        # self.model.to(self.device)
        # self.model.eval()
        
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        nltk.download('words')
        self.word_list = words.words()
        self.engine = pyttsx3.init()
        
        self.sentence_buffer = ''
        self.current_suggestions = []
        self.last_time_hand_seen = time.time()
        self.last_predicted_label = None
        self.last_prediction_time = 0
        self.prediction_duration_threshold = 2
        self.word_selected_time = 0
        self.selected_word = None

        self.label_map = label_map
        self.transform = getTransform(size=128)

    def speak_text(self, text):
        """Convert text to speech."""
        self.engine.say(text)
        self.engine.runAndWait()

    def suggest_words(self, prefix, num_suggestions=3):
        """Generate suggestions for words starting with the given prefix."""
        filtered_words = [word.upper() for word in self.word_list if word.startswith(prefix.lower())][:num_suggestions]
        return filtered_words
    
    # def thresholding(self, pil_image):
    #     """Apply image thresholding to enhance hand features."""
    #     image = np.array(pil_image)
    #     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #     binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    #     kernel = np.ones((3,3), np.uint8)
    #     binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    #     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     for cnt in contours:
    #         if cv2.contourArea(cnt) < 200:
    #             cv2.drawContours(binary, [cnt], 0, 0, -1)
    #     binary = cv2.bitwise_not(binary)
    #     cv2.imshow("Thresholded Image", binary)
    #     return Image.fromarray(binary)

    def update_sentence_buffer(self, predicted_label, button_selected=False):
        """Update the sentence buffer with new input or control commands."""
        current_time = time.time()
        if not button_selected:
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

                current_word_prefix = self.sentence_buffer.strip().split()[-1] if self.sentence_buffer.strip().split() else ''
                self.current_suggestions = self.suggest_words(current_word_prefix)

                self.last_prediction_time = current_time

        if current_time - self.last_time_hand_seen > 5.0 and not button_selected:
            if self.sentence_buffer.strip():
                self.speak_text(self.sentence_buffer)
                print("Final sentence:", self.sentence_buffer)
                self.sentence_buffer = ''
                self.current_suggestions = []
        self.last_time_hand_seen = current_time
        return self.current_suggestions

    def draw_buttons(self, frame, suggestions, x_start, y_start, height=50, width=200, vertical_space=10):
        """Draw interactive buttons for word suggestions on the frame."""
        button_list = []
        for i, word in enumerate(suggestions):
            top_left = (x_start, y_start + i * (height + vertical_space))
            bottom_right = (x_start + width, top_left[1] + height)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, word, (top_left[0] + 5, top_left[1] + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            button_list.append((top_left, bottom_right, word))
        return button_list

    def check_hand_over_button(self, hand_landmarks, buttons, frame):
        """Check if the index finger is over any button and return the associated word."""
        index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        x, y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
        for button in buttons:
            if button[0][0] <= x <= button[1][0] and button[0][1] <= y <= button[1][1]:
                return button[2]
        return None

    # def detect_and_crop_hand(self, frame):
    #     """Detect hands using MediaPipe and crop the image around the detected hand."""
    #     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     results = self.mp_hands.process(image_rgb)
    #     if results.multi_hand_landmarks:
    #         for hand_landmarks in results.multi_hand_landmarks:
    #             x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
    #             x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
    #             y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
    #             y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
    #             x_min, x_max = max(0, x_min - 50), min(frame.shape[1], x_max + 50)
    #             y_min, y_max = max(0, y_min - 50), min(frame.shape[0], y_max + 50)
    #             return frame[int(y_min):int(y_max), int(x_min):int(x_max)], True
    #     return frame, False

    def predict_sign_language(self, frame):
        """Predict the sign language label from the provided frame."""
        pil_image = Image.fromarray(frame)
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            _, predicted = torch.max(output, 1)
            predicted_label = self.label_map[predicted.item()]
        return predicted_label
    

    def process_frame(self, frame):
        """Process each frame of the video for sign language recognition and UI updates."""
        results = self.mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        found = results.multi_hand_landmarks is not None
        if found:
            cropped_frame, _ = detect_and_crop_hand(frame, self.mp_hands)
            label = self.predict_sign_language(cropped_frame)

            buttons = self.draw_buttons(frame, self.current_suggestions, int(frame.shape[1] * 0.8), 30)
            new_selected_word = None
            for hand_landmarks in results.multi_hand_landmarks:
                new_selected_word = self.check_hand_over_button(hand_landmarks, buttons, frame)
                if new_selected_word:
                    break

            # Manage word selection timing and append to sentence
            if new_selected_word:
                if new_selected_word == self.selected_word:
                    if time.time() - self.word_selected_time >= 2:
                        if ' ' in self.sentence_buffer:
                            last_space_index = self.sentence_buffer.rfind(' ')
                            self.sentence_buffer = self.sentence_buffer[:last_space_index + 1] + new_selected_word
                        else:
                            self.sentence_buffer = new_selected_word
                        self.current_suggestions = self.suggest_words(new_selected_word)
                        self.word_selected_time = 0
                else:
                    self.selected_word = new_selected_word
                    self.word_selected_time = time.time()
            else:
                self.selected_word = None
                self.word_selected_time = 0

            self.update_sentence_buffer(label, button_selected=bool(new_selected_word))
            info_text = f'Label: {label}, Sentence: {self.sentence_buffer}'
            cv2.putText(frame, info_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.draw_buttons(frame, self.current_suggestions, int(frame.shape[1] * 0.8), 30)

        cv2.imshow('Sign Language Recognition', frame)


    def run(self):
        """Main loop to capture video frames and process them."""
        cap = cv2.VideoCapture(0)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                self.process_frame(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.mp_hands.close()

# Example of usage
if __name__ == "__main__":
    model_path = 'sign_language_cnn_space_delete_good.pth'
    label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'delete', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M',
                          14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'space', 21: 'T', 22: 'U', 23: 'V', 24: 'W', 25: 'X', 26: 'Y',
                          27: 'Z'}
    recognizer = SignLanguageRecognizerWithSuggestions(model_path, label_map)
    recognizer.run()
