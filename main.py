import cv2
import pytesseract
from spellchecker import SpellChecker
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the fine-tuned model and tokenizer
model_path = "./fine-tuned-t5-ocr"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the image
image_path = './dump/slide.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Preprocess the image
# Apply thresholding
_, thresholded = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

# Save preprocessed image
cv2.imwrite('./dump/preprocessed_slide.jpg', thresholded)

# Config:
# oem 1: LSTM OCR Engine
# psm 6: Assume a single uniform block of text
# preserve_interword_spaces=1: Preserve interword spaces
config = '--oem 1 --psm 6 -c preserve_interword_spaces=1'

# Extract text from the preprocessed image
text = pytesseract.image_to_string('./dump/preprocessed_slide.jpg', lang='spa', config=config)
print(text)

# Correct spelling and common OCR mistakes
# spell = SpellChecker(language='es')
# corrected_text = ' '.join([spell.correction(word) or word for word in text.split()])

# # Use the fine-tuned model to correct OCR text
# inputs = tokenizer(corrected_text, return_tensors="pt", max_length=512, truncation=True)
# outputs = model.generate(inputs["input_ids"], max_length=512, num_beams=4, early_stopping=True)
# corrected_ocr_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# print("Corrected OCR Text:")
# print(corrected_ocr_text)