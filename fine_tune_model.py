from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from dataset_preparation import load_and_preprocess_dataset
import torch

def fine_tune_model(image_dir, text_dir, output_dir="./fine-tuned-t5-ocr"):
    tokenized_datasets, tokenizer = load_and_preprocess_dataset(image_dir, text_dir)

    # Setup
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    print("Using GPU" if torch.cuda.is_available() else "Using CPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Train
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer
    )

    trainer.train()

    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

if __name__ == "__main__":
    fine_tune_model("./dataset/train/images/", "./dataset/train/texts/")
