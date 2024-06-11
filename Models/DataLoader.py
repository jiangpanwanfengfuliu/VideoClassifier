from torch.utils.data import Dataset, DataLoader


class VideoDataset(Dataset):
    def __init__(self, data, labels) -> None:
        super(VideoDataset, self).__init__()
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]
        return sample, label
