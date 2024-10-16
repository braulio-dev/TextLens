import argparse
from utils import preprocess_image, extract_text_from_image, correct_text_with_spellchecker, correct_text_with_model, evaluate_ocr

def main(image_path, model_path, ground_truth_text=None, evaluate_metrics=False):
    # Preprocess the image
    preprocessed_image_path = preprocess_image(image_path, './dump/preprocessed_slide.jpg')

    # Extract text from the preprocessed image
    text = extract_text_from_image(image_path, lang='spa')
    # text = extract_text_from_image(preprocessed_image_path, lang='spa')

    # Correct spelling and common OCR mistakes
    corrected_text = correct_text_with_spellchecker(text, language='es')

    print("Read Text:")
    print(text)

    # Optionally evaluate OCR performance
    if evaluate_metrics and ground_truth_text:
        cer_value, wer_value, levenshtein_value = evaluate_ocr(text, ground_truth_text)
        print(f"CER: {cer_value}")
        print(f"WER: {wer_value}")
        print(f"Levenshtein Distance: {levenshtein_value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR and text correction")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image")
    parser.add_argument('--model_path', type=str, default="./fine-tuned-t5-ocr", help="Path to the fine-tuned model")
    parser.add_argument('--ground_truth_text', type=str, help="Ground truth text for evaluation")
    parser.add_argument('--evaluate_metrics', action='store_true', help="Flag to evaluate OCR metrics")

    args = parser.parse_args()
    main(args.image_path, args.model_path, args.ground_truth_text, args.evaluate_metrics)