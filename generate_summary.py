from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def generate_summary(text, model_name="t5-large"):
    # Load the pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Generate the summarized text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=1000, min_length=80, length_penalty=2.0, num_beams=5, early_stopping=False)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

if __name__ == "__main__":
    text = "Your input text here."
    summary = generate_summary(text)
    print(summary)