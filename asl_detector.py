import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import string
import pyttsx3
import mediapipe as mp
from transformers import BertTokenizer, BertForMaskedLM
from torchvision import transforms

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
            nn.Linear(256, 24)  # 24 classes
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
# 3. NLP MODEL FOR WORD SUGGESTIONS (USING PYTORCH-SPECIFIC BERT)
# ----------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

def suggest_words(prefix, topk=5):
    """Suggest top-k completions for a prefix using BERT masked LM (PyTorch version)"""
    if not prefix:
        return []
    input_text = prefix + " [MASK]"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    
    with torch.no_grad():
        logits = bert_model(**inputs).logits
    
    # Get top-k token IDs for the mask position
    predicted_token_ids = logits[0, mask_token_index].topk(topk, dim=-1).indices[0]
    
    # Decode each token ID to string (handling subwords by removing '##' prefix if present)
    suggestions = []
    for token_id in predicted_token_ids:
        token_str = tokenizer.decode([token_id])
        if token_str.startswith('##'):
            token_str = token_str[2:]  # Remove '##' for subword continuations
        suggestions.append(token_str.strip())
    
    return [prefix + s for s in suggestions]

# ----------------------
# 4. TTS ENGINE
# ----------------------
engine = pyttsx3.init()

# ----------------------
# 5. HAND DETECTION AND IMAGE PREPROCESSING
# ----------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def predict_letter(frame, hands):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if not results.multi_hand_landmarks:
        return None
    hand_landmarks = results.multi_hand_landmarks[0]

    h, w, c = frame.shape
    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
    y_coords = [lm.y * h for lm in hand_landmarks.landmark]
    x_min = max(0, int(min(x_coords) - 20))
    y_min = max(0, int(min(y_coords) - 20))
    x_max = min(w, int(max(x_coords) + 20))
    y_max = min(h, int(max(y_coords) + 20))
    if x_max <= x_min or y_max <= y_min:
        return None

    hand_crop = frame[y_min:y_max, x_min:x_max]

    # Use torchvision transforms for consistency with training
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    tensor = transform(hand_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        _, pred = torch.max(outputs, 1)
        return classes[pred.item()]


# ----------------------
# 6. MAIN LOOP
# ----------------------
def main():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    current_prefix = ""   # accumulating predicted letters
    final_sentence = ""   # chosen words
    suggestions = []

    print("Press SPACE to capture & predict a letter")
    print("Press BACKSPACE to delete last letter")
    print("Press 1-5 to select a suggestion")
    print("Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect hands and draw landmarks for visualization
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        cv2.imshow("ASL Detector", frame)

        key = cv2.waitKey(1) & 0xFF

        # SPACE = predict a new letter
        if key == ord(' '):
            letter = predict_letter(frame, hands)
            if letter is None:
                print("No hand detected. Position your hand clearly and try again.")
                continue
            current_prefix += letter.lower()  # Use lowercase for prefix
            print(f"Predicted letter: {letter}")
            print(f"Current prefix: {current_prefix}")

            # update suggestions
            suggestions = suggest_words(current_prefix, topk=5)
            if suggestions:
                print("Suggestions:")
                for i, word in enumerate(suggestions, 1):
                    print(f"{i}. {word}")

        # BACKSPACE = delete last letter
        elif key == 8:
            if current_prefix:
                current_prefix = current_prefix[:-1]
                print(f"Current prefix: {current_prefix}")
                suggestions = suggest_words(current_prefix, topk=5)
                if suggestions:
                    print("Suggestions:")
                    for i, word in enumerate(suggestions, 1):
                        print(f"{i}. {word}")

        # NUMBER KEYS = select word
        elif key in [ord(str(i)) for i in range(1, 6)]:
            idx = int(chr(key)) - 1
            if idx < len(suggestions):
                chosen_word = suggestions[idx]
                final_sentence += chosen_word + " "
                current_prefix = ""  # reset
                suggestions = []
                print(f"Chosen word: {chosen_word}")
                print(f"Final sentence: {final_sentence}")
                engine.say(chosen_word)
                engine.runAndWait()

        # Q = quit
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()