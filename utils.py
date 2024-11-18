import cv2
import pytesseract
from spellchecker import SpellChecker
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import re
import torch
from PIL import Image, ImageEnhance
from torchvision import transforms

model_path = "./ocr-10.16.24/"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def cer(s1, s2):
    return levenshtein_distance(s1, s2) / len(s2)

def wer(s1, s2):
    s1_words = s1.split()
    s2_words = s2.split()
    return levenshtein_distance(s1_words, s2_words) / len(s2_words)

def evaluate_ocr(text, ground_truth_text):
    trimmed = ' '.join(text.replace('\n', ' ').split())
    cer_value = cer(trimmed, ground_truth_text)
    wer_value = wer(trimmed, ground_truth_text)
    levenshtein_value = levenshtein_distance(trimmed, ground_truth_text)
    return cer_value, wer_value, levenshtein_value

def normalize_spanish_accents(text):
    replacements = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'ñ': 'n', 'Ñ': 'N'
    }
    for accented_char, normal_char in replacements.items():
        text = text.replace(accented_char, normal_char)
    return text

def extract_text_from_image(image_path, lang, config='--oem 1 --psm 6 -c preserve_interword_spaces=1'):
    processed_image = Image.open(image_path)

    # Grayscale
    processed_image = processed_image.convert('L')

    # Increase contrast
    enhancer = ImageEnhance.Contrast(processed_image)
    processed_image = enhancer.enhance(2)

    # Resizing
    processed_image = processed_image.resize((processed_image.width * 3, processed_image.height * 3))

    # Write the processed image to a file
    processed_image.save("./dump/processed_image.png")

    text = pytesseract.image_to_string(processed_image, lang=lang, config=config)
    text = normalize_spanish_accents(text)
    return text.strip()

def preprocess_text(text):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512  # Ensure this matches the model's max length
    )
    return inputs

def get_model_output(image_path, lang):
    extracted_text = extract_text_from_image(image_path, lang)
    if not extracted_text:
        return "No text extracted from the image"
    
    inputs = preprocess_text(extracted_text)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    predicted_class = torch.argmax(outputs.logits, dim=-1)
    return tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)