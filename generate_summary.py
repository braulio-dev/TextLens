from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

def generate_summary(text, model_name="mrm8488/bert2bert_shared-spanish-finetuned-summarization"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    
    length = int(len(text.split()) * 0.75)
    min_length = int(max(10, length * 0.4))
    summary = summarizer(text, max_length=length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    text = "Your input text here."
    summary = generate_summary(text)
    print(summary)