import argparse
from datasets import load_dataset
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# accepts the following command line arguments:
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_train_samples', type=int, default=-1)
    parser.add_argument('--max_eval_samples', type=int, default=-1)
    parser.add_argument('--max_predict_samples', type=int, default=-1)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--model_path', type=str, default='bert-base-uncased')
    args = parser.parse_args()
    return args


def preprocess_data(args):
    wandb.init(project='ex1', config=vars(args))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    ds = load_dataset("nyu-mll/glue", "mrpc")

    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

    tokenized_ds = ds.map(tokenize_function, batched=True)

    if args.max_train_samples != -1:
        tokenized_ds["train"] = tokenized_ds["train"].select(range(args.max_train_samples))
    if args.max_eval_samples != -1:
        tokenized_ds["validation"] = tokenized_ds["validation"].select(range(args.max_eval_samples))
    if args.max_predict_samples != -1:
        tokenized_ds["test"] = tokenized_ds["test"].select(range(args.max_predict_samples))

    return tokenizer, ds, tokenized_ds

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def fine_tune(args, tokenized_ds, ds, tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=2)
    training_args = TrainingArguments(
        output_dir="output",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        # evaluation_strategy="epoch",
        learning_rate=args.lr,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=1,
        report_to=["wandb"],
        save_strategy="no"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer = tokenizer) if args.do_train else None,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        compute_metrics=compute_metrics
    )

    if args.do_train:
        trainer.train()
        checkpoint_name = f"model_e{args.num_train_epochs}_lr{args.lr}_bs{args.batch_size}"
        trainer.save_model(checkpoint_name)

        val_results = trainer.evaluate()
        with open("res.txt", "a") as res_file:
            res_file.write(
                f"epoch_num: {args.num_train_epochs}, lr: {args.lr}, batch_size: {args.batch_size}, eval_acc: {val_results['eval_accuracy']:.4f}\n"
            )

    if args.do_predict:
        model.eval()
        predictions = trainer.predict(tokenized_ds["test"])
        preds = np.argmax(predictions.predictions, axis=1)

        pred_filename = f"predictions.txt"
        with open(pred_filename, "w", encoding="utf-8") as f:
            for example, pred in zip(ds["test"], preds):
                f.write(f"{example['sentence1']}###{example['sentence2']}###{pred}\n")

def main():
   args = parse_args()
   tokenizer, ds, tokenized_ds = preprocess_data(args)
   fine_tune(args, tokenized_ds, ds, tokenizer)

if __name__ == "__main__":
    main()
