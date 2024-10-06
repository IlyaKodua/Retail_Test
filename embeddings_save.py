from torch.utils.data import DataLoader
import warnings
import torch
from custom_dataset import RetailDataset
from utils import get_embeddings, top_n
import ruclip
import pickle
warnings.filterwarnings('ignore')

root_dir = "dataset"

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = ruclip.load('ruclip-vit-base-patch16-384', device=device)
model = model.visual.float().to(device)

dataset = RetailDataset(root_dir = root_dir, preprocess = preprocess.image_transform)

dataloader = DataLoader(dataset, batch_size=4, num_workers=8, shuffle=True)

embeddings = get_embeddings(model, dataloader, device)

with open('embeddings_dict.pickle', 'wb') as f:
    pickle.dump(embeddings, f)

top1 = top_n(embeddings[0], embeddings[1], 1)
top5 = top_n(embeddings[0], embeddings[1], 5)

print("top1: ", top1)
print("top5: ", top5)