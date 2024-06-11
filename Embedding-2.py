import os
import pandas as pd
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from Models.VideoClassfier import VideoClassifier
from Models.DataLoader import VideoDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vids_seq(sequence_length, input_path):
    """Load Video Sequences"""
    file_names = []
    sequences = []
    file_list = os.listdir(input_path)
    for file_name in file_list:
        video_file = pd.read_csv(input_path + file_name, index_col=None)
        sequence = video_file.iloc[:, 2].to_list()
        file_names.append(file_name)
        sequences.append(sequence)

    sequences = [
        sequence + [0] * (sequence_length - len(sequence)) for sequence in sequences
    ]
    return file_names, sequences


def load_vids_csv(data_name, label_name):
    labels = pd.read_csv(label_name).class_label.tolist()
    sequences = torch.Tensor(pd.read_csv(data_name).to_numpy())
    return sequences, labels


def progress(i, total_epoches, loss):
    """Show Current Progress"""
    print(f"Epoch {i}/{total_epoches}, Loss: {loss:.4f}")


def score(model, X_test, Y_test, labels_enc):
    """Calculate Precision,Recall and Accuracy"""
    Y_pred = model(X_test).argmax(axis=1)
    cm = confusion_matrix(Y_test, Y_pred)
    accs = cm.diagonal() / cm.sum(axis=1)
    precs = precision_score(Y_test, Y_pred, average=None)
    recalls = recall_score(Y_test, Y_pred, average=None)

    rates, nums = [], []
    labels = torch.unique(Y_test)
    labels.sort()
    for label in labels:
        indices = torch.where(Y_test == label)[0]  # 获取属于当前类别的样本索引
        rates.append(len(indices) / len(Y_pred))
        nums.append(len(indices))
    labels = labels_enc.inverse_transform(labels)
    print("*" * 80)
    print(
        "{:20s}{:10}{:10}{:10}{:10}{:10}".format(
            "name", "acc", "prec", "recall", "rate", "num"
        )
    )
    for label, acc, prec, recall, rate, num in zip(
        labels, accs, precs, recalls, rates, nums
    ):
        print(
            "{:20s}{:<10.3f}{:<10.3f}{:<10.3f}{:<10.3f}{:<10}".format(
                label, acc, prec, recall, rate, num
            )
        )
    print("*" * 80)
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average="macro")
    recall = recall_score(Y_test, Y_pred, average="macro")
    print("Accuracy Score:", accuracy)
    print("Precision Score:", precision)
    print("Recall Score:", recall)
    print("*" * 80)


if __name__ == "__main__":
    """1.Config Device"""

    """2.Load Data"""
    # Loads Videos' Sequences
    data_name = "3-datasets/train-data-v2.csv"
    label_name = "3-datasets/train-labels-v2.csv"
    sequences, labels = load_vids_csv(data_name, label_name)

    sequences = sequences.reshape((sequences.shape[0], sequences.shape[1], 1))
    print(sequences.shape)
    labels_enc = LabelEncoder()
    labels = labels_enc.fit_transform(labels)
    labels = torch.Tensor(labels).long()
    X_train, X_test, Y_train, Y_test = train_test_split(
        sequences, labels, test_size=0.1
    )
    input_dim = X_train.shape[1]
    output_dim = torch.unique(Y_train).shape[0]
    print("input_dim:", input_dim)
    print("output_dim:", output_dim)
    print("sequences shape:", *sequences.shape)

    # Get Dataset
    dataset = VideoDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    """3.Train Model"""
    model_params = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "in_channels": 1,
        "out_channels": 10,
        "kernel_size": 3,
        "stride": 3,
        "num_layers": 1,
    }
    model_name = "Video-Classifier-Model-2.pth"
    model = VideoClassifier(**model_params)

    # hook
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    model.encoder.register_forward_hook(get_activation("embedding"))

    if model_name in os.listdir("5-weights"):
        model_path = os.path.join(os.getcwd(), "5-weights", model_name)
        model.load_state_dict(
            torch.load(
                model_path,
                map_location=torch.device("cpu"),
            )
        )

    epoch = 10
    lr = 0.00001
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr)
    if device.type == "cuda":
        model = model.to(device)

    """4.Evaluate Model"""
    model.to(device=torch.device("cpu"))
    data = []
    for i in range(sequences.shape[0] // 64 + 1):
        # score(model, X, Y, labels_enc)
        model(sequences[i * 64 : (i + 1) * 64, :])
        data.append(activation["embedding"])
    torch.save(torch.concat(data), "4-graphs/temp/train-data-v2.pth")
