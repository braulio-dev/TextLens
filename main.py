import argparse
from utils import get_model_output, evaluate_ocr
from generate_summary import generate_summary

def main(image_path, model_name, lang='spa', ground_truth_text=None, evaluate_metrics=False):
    text = get_model_output(image_path, lang)

    print("Read Text:")
    print(text)

    print("\nSummary:")
    print(generate_summary(text, model_name))

    if evaluate_metrics and ground_truth_text:
        cer_value, wer_value, levenshtein_value = evaluate_ocr(text, ground_truth_text)
        print(f"CER: {cer_value}")
        print(f"WER: {wer_value}")
        print(f"Levenshtein Distance: {levenshtein_value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR and text correction")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image")
    parser.add_argument('--model_name', type=str, default="mrm8488/bert2bert_shared-spanish-finetuned-summarization", help="Name of the pre-trained model")
    parser.add_argument('--lang', type=str, default='spa', help="Language for OCR")
    parser.add_argument('--ground_truth_text', type=str, help="Ground truth text for evaluation")
    parser.add_argument('--evaluate_metrics', action='store_true', help="Flag to evaluate OCR metrics")

    args = parser.parse_args()
    main(args.image_path, args.model_name, args.lang, args.ground_truth_text, args.evaluate_metrics)