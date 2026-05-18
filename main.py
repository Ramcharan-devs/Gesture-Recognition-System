import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
from gtts import gTTS
import os
import threading
from playsound import playsound

# MediaPipe initialization
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.7)

# Emoji mapping
EMOJIS = {
    "smile": "😄",
    "sad": "😢",
    "surprise": "😲",
    "thumbs up": "👍",
    "peace": "✌",
    "fist": "✊",
    "open palm": "🖐",
    "calling": "🤙",
    "love": "💖",
    "thumbs down": "👎",
    "horns": "🤘",
    "no face": "🚫",
    "no hand": "❓"
}

# Action labels for gestures
ACTIONS = {
    "thumbs up": "Drinking Water",
    "peace": "Peace Mode Activated",
    "fist": "Power Gesture",
    "calling": "Calling Gesture",
    "horns": "Rock On",
    "thumbs down": "Disapproval",
    "love": "Love You Gesture Detected",
    "open palm": "Open Palm",
    "no hand": "No hand detected"
}

# Function to speak text using gTTS without freezing GUI
def speak(text):
    def _speak():
        tts = gTTS(text=text, lang='en')
        filename = "temp_audio.mp3"
        tts.save(filename)
        playsound(filename)
        os.remove(filename)
    threading.Thread(target=_speak, daemon=True).start()

# Gesture and expression detection
def detect(frame):
    h, w, _ = frame.shape
    results = {"face": None, "hand": None}
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face detection
    face_result = face_mesh.process(rgb)
    if face_result.multi_face_landmarks:
        lm = face_result.multi_face_landmarks[0].landmark
        top_lip = lm[13].y
        bottom_lip = lm[14].y
        mouth_gap = (bottom_lip - top_lip) * h

        if mouth_gap > 20:
            results["face"] = "surprise"
        elif mouth_gap < 5:
            results["face"] = "sad"
        else:
            results["face"] = "smile"
    else:
        results["face"] = "no face"

    # Hand gesture detection
    hand_result = hands.process(rgb)

    if hand_result.multi_hand_landmarks and len(hand_result.multi_hand_landmarks) == 2:
        lm1 = hand_result.multi_hand_landmarks[0].landmark
        lm2 = hand_result.multi_hand_landmarks[1].landmark

        index_tip_1 = np.array([lm1[8].x, lm1[8].y])
        index_tip_2 = np.array([lm2[8].x, lm2[8].y])
        thumb_tip_1 = np.array([lm1[4].x, lm1[4].y])
        thumb_tip_2 = np.array([lm2[4].x, lm2[4].y])

        index_dist = np.linalg.norm(index_tip_1 - index_tip_2)
        thumb_dist = np.linalg.norm(thumb_tip_1 - thumb_tip_2)

        if index_dist < 0.1 and thumb_dist < 0.1:
            results["hand"] = "love"
        else:
            results["hand"] = "open_palm"

    elif hand_result.multi_hand_landmarks:
        lm = hand_result.multi_hand_landmarks[0].landmark

        fingers = {
            "thumb": lm[4].x < lm[3].x,
            "index": lm[8].y < lm[6].y,
            "middle": lm[12].y < lm[10].y,
            "ring": lm[16].y < lm[14].y,
            "pinky": lm[20].y < lm[18].y
        }

        extended_count = sum(fingers.values())
        distance_thumb_index = np.linalg.norm(
            np.array([lm[4].x, lm[4].y]) - np.array([lm[8].x, lm[8].y])
        )

        if distance_thumb_index < 0.03 and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
            results["hand"] = "love"
        elif fingers["thumb"] and not any([fingers["index"], fingers["middle"], fingers["ring"], fingers["pinky"]]):
            results["hand"] = "thumbs_up"
        elif not fingers["thumb"] and fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
            results["hand"] = "peace"
        elif not any(fingers.values()):
            wrist_y = lm[0].y
            thumb_y = lm[4].y
            if thumb_y > wrist_y:
                results["hand"] = "thumbs_down"
            else:
                results["hand"] = "fist"
        elif fingers["thumb"] and not fingers["index"] and not fingers["middle"] and not fingers["ring"] and fingers["pinky"]:
            results["hand"] = "calling"
        elif not fingers["thumb"] and fingers["index"] and not fingers["middle"] and not fingers["ring"] and fingers["pinky"]:
            results["hand"] = "horns"
        elif extended_count >= 4:
            results["hand"] = "open_palm"
        else:
            results["hand"] = "open_palm"
    else:
        results["hand"] = "no hand"

    return results

# GUI App class
class GestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ram's Gesture Recognition")
        self.root.geometry("800x600")

        self.label = tk.Label(root)
        self.label.pack()

        self.emoji_label = tk.Label(root, font=("Arial", 24))
        self.emoji_label.pack()

        self.action_label = tk.Label(root, font=("Arial", 16), fg="blue")
        self.action_label.pack()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot access webcam.")
            return

        # Track last sentence spoken to avoid repetition
        self.last_sentence = ""

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.action_label.config(text="No webcam frame detected!")
            self.root.after(60, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        result = detect(frame)

        face_emoji = EMOJIS.get(result["face"], "")
        hand_emoji = EMOJIS.get(result["hand"], "")
        action_text = ACTIONS.get(result["hand"], "")

        # Create full sentence for TTS
        sentence = f"Face detected: {result['face']} {face_emoji}. Hand gesture: {result['hand']} {hand_emoji}. Action: {action_text}."

        # Speak only if sentence changed
        if sentence != self.last_sentence:
            speak(sentence)
            self.last_sentence = sentence

        # Update GUI
        self.emoji_label.config(
            text=f"Face: {result['face']} {face_emoji} | Hand: {result['hand']} {hand_emoji}"
        )
        self.action_label.config(text=action_text)

        # Display webcam frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)

        self.root.after(30, self.update_frame)

    def on_closing(self):
        self.cap.release()
        hands.close()
        face_mesh.close()
        cv2.destroyAllWindows()
        self.root.destroy()

# Main loop
if __name__ == "__main__":
    root = tk.Tk()
    app = GestureApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()