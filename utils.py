import cv2
import pytesseract
from spellchecker import SpellChecker
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np
import re

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

def preprocess_image(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(output_path, image)
    return output_path

def clean_text(text):
    # Remove unwanted characters and trailing spaces, but keep essential punctuation
    text = re.sub(r'[^\w\s.,;:!?\'"-]', '', text)  # Remove non-alphanumeric characters except spaces and essential punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip()  # Remove leading and trailing spaces

def extract_text_from_image(image_path, lang='spa', config='--oem 1 --psm 6 -c preserve_interword_spaces=1'):
    text = pytesseract.image_to_string(image_path, lang=lang, config=config)
    return clean_text(text)  # Clean the extracted text before returning

def evaluate_ocr(text, ground_truth_text):
    trimmed = ' '.join(text.replace('\n', ' ').split())
    cer_value = cer(trimmed, ground_truth_text)
    wer_value = wer(trimmed, ground_truth_text)
    levenshtein_value = levenshtein_distance(trimmed, ground_truth_text)
    return cer_value, wer_value, levenshtein_value