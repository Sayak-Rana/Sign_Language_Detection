import cv2
import torch
import torch.nn as nn
import numpy as np
import string
import pyttsx3
import mediapipe as mp
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer, BertForMaskedLM
import string as str_mod  # Rename to avoid conflict with imported string

# ----------------------
# 1. CNN MODEL
# ----------------------
class SignLanguageCNN(nn.Module):
    def __init__(self):
        super(SignLanguageCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 24)  # 24 classes (A-Y, excluding J, Z)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# ----------------------
# 2. LOAD MODEL
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLanguageCNN().to(device)
model.load_state_dict(torch.load("sign_language_cnn.pth", map_location=device))
model.eval()

# 24 valid letters (no J, Z)
classes = [c for c in string.ascii_uppercase if c not in ['J', 'Z']]

# ----------------------
# 3. NLP MODEL FOR WORD SUGGESTIONS
# ----------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

def suggest_words(prefix, topk=5):
    """Suggest top-k completions for a prefix using BERT masked LM"""
    if not prefix:
        return []
    input_text = prefix + "[MASK]"  # Removed space before [MASK]
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    
    with torch.no_grad():
        logits = bert_model(**inputs).logits
    
    # Get top-50 token IDs for the mask position to filter later
    predicted_token_ids = logits[0, mask_token_index].topk(50, dim=-1).indices[0]
    
    # Decode each token ID to string (handling subwords)
    suggestions = []
    for token_id in predicted_token_ids:
        token_str = tokenizer.decode([token_id])
        if token_str.startswith('##'):
            token_str = token_str[2:]  # Remove '##' for subword continuations
        token_str = token_str.strip()
        # Filter to alphabetic suggestions only
        if token_str and all(c.isalpha() for c in token_str):
            suggestions.append(prefix + token_str)
            if len(suggestions) == topk:
                break
    
    return suggestions

# ----------------------
# 4. TTS ENGINE
# ----------------------
engine = pyttsx3.init()

# ----------------------
# 5. HAND DETECTION AND IMAGE PREPROCESSING
# ----------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Transform to match sign_mnist training
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Match training normalization
])

def predict_letter(frame, hands):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if not results.multi_hand_landmarks:
        return None, None
    hand_landmarks = results.multi_hand_landmarks[0]
    h, w, c = frame.shape
    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
    y_coords = [lm.y * h for lm in hand_landmarks.landmark]
    x_min = max(0, int(min(x_coords) - 20))
    y_min = max(0, int(min(y_coords) - 20))
    x_max = min(w, int(max(x_coords) + 20))
    y_max = min(h, int(max(y_coords) + 20))
    if x_max <= x_min or y_max <= y_min:
        return None, None
    hand_crop = frame[y_min:y_max, x_min:x_max]
    # Apply transform to match training
    tensor = transform(hand_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        _, pred = torch.max(outputs, 1)
        return classes[pred.item()], hand_crop

# ----------------------
# 6. MAIN LOOP
# ----------------------
def main():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
    
    current_prefix = ""
    final_sentence = ""
    pred_text = ""
    old_text = ""
    count_frames = 0
    flag = False
    suggestions = []
    prefix_changed = False

    print("Press 'c' to start continuous prediction")
    print("Press SPACE to capture & predict a single letter")
    print("Press BACKSPACE to delete last letter")
    print("Press 1-5 to select a suggestion")
    print("Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect hands and draw landmarks
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        hand_crop = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Draw bounding box
                h, w, c = frame.shape
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                x_min = max(0, int(min(x_coords) - 20))
                y_min = max(0, int(min(y_coords) - 20))
                x_max = min(w, int(max(x_coords) + 20))
                y_max = min(h, int(max(y_coords) + 20))
                if x_max > x_min and y_max > y_min:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    hand_crop = frame[y_min:y_max, x_min:x_max]

        # Create blackboard for text display
        blackboard = np.zeros(frame.shape, dtype=np.uint8)
        cv2.putText(blackboard, "Predicted: " + pred_text, (30, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
        cv2.putText(blackboard, "Sentence: " + final_sentence + current_prefix, (30, 80), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 127))
        
        # Display suggestions
        pos = 120
        if suggestions:
            cv2.putText(blackboard, "Suggestions:", (30, pos), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
            pos += 40
            for i, sug in enumerate(suggestions):
                cv2.putText(blackboard, f"{i+1}. {sug}", (30, pos), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 255))
                pos += 30

        # Display processed hand image
        display_img = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY) if hand_crop is not None else np.zeros((28, 28), dtype=np.uint8)
        display_img = cv2.resize(display_img, (100, 100))  # Enlarge for visibility
        cv2.imshow("Processed Hand", display_img)

        # Show main feed and blackboard
        res = np.hstack((frame, blackboard))
        cv2.imshow("ASL Detector", res)

        # Continuous prediction logic
        if flag and hand_crop is not None:
            letter, _ = predict_letter(frame, hands)
            if letter is not None:
                old_text = pred_text
                pred_text = letter
                if old_text == pred_text:
                    count_frames += 1
                else:
                    count_frames = 0
                if count_frames > 20:
                    current_prefix += pred_text.lower()
                    count_frames = 0
                    prefix_changed = True
                    print(f"Predicted letter: {pred_text}")
                    print(f"Current prefix: {current_prefix}")

        key = cv2.waitKey(1) & 0xFF

        # SPACE = single prediction
        if key == ord(' '):
            letter, _ = predict_letter(frame, hands)
            if letter is None:
                print("No hand detected. Position your hand clearly and try again.")
                continue
            current_prefix += letter.lower()
            pred_text = letter
            prefix_changed = True
            print(f"Predicted letter: {letter}")
            print(f"Current prefix: {current_prefix}")

        # 'c' = start continuous prediction
        elif key == ord('c'):
            flag = True
            print("Continuous prediction started")

        # BACKSPACE = delete last letter
        elif key == 8:
            if current_prefix:
                current_prefix = current_prefix[:-1]
                pred_text = ""
                prefix_changed = True
                print(f"Current prefix: {current_prefix}")

        # NUMBER KEYS = select word
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            idx = key - ord('1')
            if idx < len(suggestions):
                chosen_word = suggestions[idx]
                final_sentence += chosen_word + " "
                current_prefix = ""
                suggestions = []
                pred_text = ""
                prefix_changed = True
                print(f"Chosen word: {chosen_word}")
                print(f"Final sentence: {final_sentence}")
                engine.say(chosen_word)
                engine.runAndWait()

        # Q = quit
        elif key == ord('q'):
            break

        # Update suggestions if prefix changed
        if prefix_changed:
            suggestions = suggest_words(current_prefix, topk=5)
            if suggestions:
                print("Suggestions:")
                for i, word in enumerate(suggestions, 1):
                    print(f"{i}. {word}")
            prefix_changed = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()