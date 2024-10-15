import os
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

    # Iterate over the image files
    for image_file in os.listdir(image_dir):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            # Extract the base name without extension
            base_name = os.path.splitext(image_file)[0]
            
            # Construct the corresponding text file path
            text_file = os.path.join(text_dir, base_name + ".txt")
            
            # Read the true text from the text file
            true_text = read_file_with_encodings(text_file)
            
            # Read the image file
            image_path = os.path.join(image_dir, image_file)
            ocr_text = pytesseract.image_to_string(image_path, lang='spa')
            
            # Append the extracted text and true text to the data
            data["ocr_text"].append(ocr_text)
            data["true_text"].append(true_text)
    
    # Create a Hugging Face dataset
    dataset = Dataset.from_dict(data)
    
    # Split the dataset into training and validation sets
    if len(dataset) > 1:
        dataset = dataset.train_test_split(test_size=0.1)
    else:
        dataset = DatasetDict({"train": dataset, "test": dataset})
    
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    # Tokenize the dataset
    def preprocess_function(examples):
        inputs = examples['ocr_text']
        model_inputs = tokenizer(inputs, max_length=512, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['true_text'], max_length=512, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    return tokenized_datasets, tokenizer