import cv2
import torch
from PIL import Image
import mediapipe as mp
import time
import random
from utils import getTransform, load_model, detect_and_crop_hand


class SignLanguageGame:
    def __init__(self, model_path, label_map):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(model_path, self.device)
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        self.label_map = label_map
        self.size = 128
        self.transform = getTransform(self.size)

    # def thresholding(self, pil_image):
    #     if pil_image is None:
    #         return pil_image  # Return None or handle error appropriately

    #     image = np.array(pil_image)
    #     if image.size == 0:
    #         return pil_image  # Return original or handle error

    #     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #     binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                     cv2.THRESH_BINARY_INV, 11, 2)
    #     kernel = np.ones((3, 3), np.uint8)
    #     binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    #     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     for cnt in contours:
    #         if cv2.contourArea(cnt) < 200:
    #             cv2.drawContours(binary, [cnt], 0, 0, -1)
    #     binary = cv2.bitwise_not(binary)
    #     cv2.imshow("Thresholded Image", binary)
    #     return Image.fromarray(binary)

    # def detect_and_crop_hand(self, frame):
    #     """Detect hands using MediaPipe and crop the image around the detected hand."""
    #     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     results = self.mp_hands.process(image_rgb)
    #     if results.multi_hand_landmarks:
    #         for hand_landmarks in results.multi_hand_landmarks:
    #             # Assuming only one hand per image for simplicity
    #             x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
    #             x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
    #             y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
    #             y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
    #             x_min, x_max = max(0, x_min - 50), min(frame.shape[1], x_max + 50)
    #             y_min, y_max = max(0, y_min - 50), min(frame.shape[0], y_max + 50)
    #             return frame[int(y_min):int(y_max), int(x_min):int(x_max)], True
    #     return frame, False


    def predict_sign_language(self, frame):
        if frame is None or frame.size == 0:
            return None

        pil_image = Image.fromarray(frame)
        if pil_image is None:
            return None

        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            _, predicted = torch.max(output, 1)
            return self.label_map[predicted.item()]
        

    def display_image_for_duration(self, image_path, duration, window_name):
        image = cv2.imread(image_path)
        if image is not None:
            cv2.imshow(window_name, image)
            cv2.waitKey(duration * 1000)
            cv2.destroyWindow(window_name)
        else:
            print("Image not found")


    def run(self):
        score = 0
        letters = list(self.label_map.values())
        cap = cv2.VideoCapture(0)
        target_letter = random.choice(letters)
        last_time = time.time()
        last_label = None
        label_stability_duration = 2 

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                cv2.putText(frame, f"Show sign for: {target_letter}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Score: {score}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

                # Detect and crop hand
                cropped_frame, found = detect_and_crop_hand(frame, self.mp_hands)
                if found:
                    current_label = self.predict_sign_language(cropped_frame)
                    cv2.putText(frame, f"Predicted Label: {current_label}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    current_time = time.time()

                    if current_label == last_label:
                        if (current_time - last_time) >= label_stability_duration:
                            if current_label == target_letter:
                                score += 10 
                                cv2.putText(frame, "Correct!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            else:
                                score -= 5  
                                cv2.putText(frame, "Incorrect!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                                correct_img_path = f"game/{target_letter}.jpeg"
                                correct_img = cv2.imread(correct_img_path)
                                if correct_img is not None:
                                   self.display_image_for_duration(correct_img_path, 5, "Correct Image")
                            
                            target_letter = random.choice(letters)
                            last_label = None
                            cv2.waitKey(5000)
                    else:
                        last_label = current_label
                        last_time = current_time

                cv2.imshow('Sign Language Game', frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.mp_hands.close()


if __name__ == "__main__":
#     # Assuming 'SignLanguageCNN' is already defined elsewhere
    label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'delete', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'space', 21: 'T', 22: 'U', 23: 'V', 24: 'W', 25: 'X', 26: 'Y', 27: 'Z'}
    game = SignLanguageGame('sign_language_cnn_space_delete_good.pth', label_map)
    game.run()
