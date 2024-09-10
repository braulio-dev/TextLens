import cv2
import pytesseract
import markdown
from spellchecker import SpellChecker
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Capture image from the first connected camera
cap = cv2.VideoCapture(2)
ret, frame = cap.read()

# Save the captured image
if ret:
    cv2.imwrite('./dump/slide.jpg', frame)

cap.release()

# Load the image
img = cv2.imread('./dump/slide.jpg', cv2.IMREAD_GRAYSCALE)

# Preprocessing: Resize, Thresholding
resized = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
blurred = cv2.GaussianBlur(resized, (5, 5), 0)
thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Save preprocessed image
cv2.imwrite('./dump/preprocessed_slide.jpg', thresholded)

# Config:
# oem 1: LSTM OCR Engine
# psm 6: Assume a single uniform block of text
# preserve_interword_spaces=1: Preserve interword spaces
config = '--oem 1 --psm 6 -c preserve_interword_spaces=1'

# Extract text from the preprocessed image
text = pytesseract.image_to_string('./dump/preprocessed_slide.jpg', lang='spa', config=config)

# Correct spelling and common OCR mistakes
spell = SpellChecker(language='es')
corrected_text = ' '.join([spell.correction(word) or word for word in text.split()])

# Generate the summarized text
tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
inputs = tokenizer.encode("summarize: " + corrected_text, return_tensors="pt", max_length=512, truncation=True)

summary_ids = model.generate(inputs, max_length=150, min_length=80, length_penalty=5.0, num_beams=2, early_stopping=True)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)