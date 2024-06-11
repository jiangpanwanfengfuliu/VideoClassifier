from transformers import AutoModelForSequenceClassification as AMFSC
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import os

os.environ["TORCH_USE_CUDA_DSA"] = "1"


def load_seqs():
    labels_enc = LabelEncoder()
    X = pd.read_csv("3-datasets/train-data.csv")
    Y = pd.read_csv("3-datasets/train-labels.csv").class_label.to_list()
    X = torch.Tensor(X.to_numpy())
    print(X.max())
    X = (X.view(X.shape[0], -1, 10) // 200).mean(dim=2).long()
    Y = labels_enc.fit_transform(Y)
    Y = torch.Tensor(Y).long()
    return labels_enc, X, Y


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx):
        texts = self.texts[idx]
        label = self.labels[idx]

        input_ids = texts
        attention_mask = torch.ones(texts.shape[0], dtype=int)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label,
        }

    def __len__(self):
        return len(self.texts)


labels_enc, X, Y = load_seqs()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
num_labels = torch.unique(Y).shape[0]
train_dataset = VideoDataset(X_train, Y_train)
val_dataset = VideoDataset(X_test, Y_test)
# for data in train_dataset:
#     print(data["input_ids"].shape,data["labels"].shape,data["attention_mask"].shape)


model_name = "bert-base-cased"

model = AMFSC.from_pretrained(model_name, num_labels=num_labels)


from transformers import Trainer, TrainingArguments

batch_size = 64
logging_steps = X.shape[0] // batch_size
model_name = f"{model_name}-finetuned-emotion"
training_args = TrainingArguments(
    output_dir=model_name,
    num_train_epochs=10,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps,
    push_to_hub=False,
    log_level="error",
)


from transformers import Trainer


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
trainer.train()
