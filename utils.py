import cv2
import pytesseract
from spellchecker import SpellChecker
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np

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
    _, thresholded = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    cv2.imwrite(output_path, thresholded)
    return output_path

def extract_text_from_image(image_path, lang='spa', config='--oem 1 --psm 6 -c preserve_interword_spaces=1'):
    return pytesseract.image_to_string(image_path, lang=lang, config=config)

def correct_text_with_spellchecker(text, language='es'):
    spell = SpellChecker(language=language)
    return ' '.join([spell.correction(word) or word for word in text.split()])

def correct_text_with_model(text, model_path):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=512, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_ocr(text, ground_truth_text):
    cer_value = cer(text, ground_truth_text)
    wer_value = wer(text, ground_truth_text)
    levenshtein_value = levenshtein_distance(text, ground_truth_text)
    return cer_value, wer_value, levenshtein_value