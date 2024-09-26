import cv2
import pytesseract
import markdown
from spellchecker import SpellChecker
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from generate_summary import generate_summary

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

# Generate the summarized text using the fine-tuned model
summary = generate_summary(corrected_text, model_dir="./fine-tuned-t5")
print(summary)