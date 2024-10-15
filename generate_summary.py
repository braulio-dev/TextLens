from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def generate_summary(text, model_dir="./fine-tuned-t5-ocr"):
    # Load the fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    # Generate the summarized text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=80, length_penalty=5.0, num_beams=2, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    text = "Your input text here."
    summary = generate_summary(text)
    print(summary)