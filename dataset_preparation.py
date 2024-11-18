import os
import re
import pytesseract
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

def read_file_with_encodings(file_path, encodings=['utf-8', 'latin-1', 'iso-8859-1']):
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read().strip()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Unable to decode the file {file_path} with provided encodings.")

def load_and_preprocess_dataset(image_dir, text_dir):
    data = {"ocr_text": [], "true_text": []}

    # Get all images
    for image_file in os.listdir(image_dir):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            # Get name of image
            base_name = os.path.splitext(image_file)[0]
            trimmed_base_name = re.sub(r"\s*\(\d+\)$", "", base_name)

            # Get truth text
            text_file = os.path.join(text_dir, trimmed_base_name + ".txt")
            true_text = read_file_with_encodings(text_file)
            
            # Extract text
            image_path = os.path.join(image_dir, image_file)
            ocr_text = pytesseract.image_to_string(image_path, lang='spa')
            
            # Results
            data["ocr_text"].append(ocr_text)
            data["true_text"].append(true_text)
    
    dataset = Dataset.from_dict(data)
    
    if len(dataset) > 1:
        dataset = dataset.train_test_split(test_size=0.1)
    else:
        dataset = DatasetDict({"train": dataset, "test": dataset})
    
    # Tokenize 
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    def preprocess_function(examples):
        inputs = examples['ocr_text']
        model_inputs = tokenizer(inputs, padding='max_length', max_length=512, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['true_text'], padding='max_length', max_length=512, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    return tokenized_datasets, tokenizer