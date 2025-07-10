import torch
from torch.utils.data import Dataset

# --- Dataset ---
class MIDIDataset(Dataset):
    def __init__(self, embeddings):
        self.data = torch.tensor(embeddings, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]