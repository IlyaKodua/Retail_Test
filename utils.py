import numpy as np
import torch
from tqdm import tqdm

def top_n(embeddings1 : np.array, embeddings2 : np.array, n : int) -> float:
    """
    Finds the top-n most similar embeddings for each embedding in the first set.

    Args:
        embeddings1 (numpy.ndarray): Array of embeddings from the first set.
        embeddings2 (numpy.ndarray): Array of embeddings from the second set.
        n (int): Number of top similar embeddings to find.

    Returns:
        float: Top-N accuracy metric
    """

    embeddings1_normalized = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2_normalized = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)

    similarity_matrix = np.dot(embeddings1_normalized, embeddings2_normalized.T)

    top_n_indices = np.argpartition(-similarity_matrix, n, axis=1)[:, :n]  # Descending order

    mask_n = np.arange(len(top_n_indices))[:,None] == top_n_indices


    return mask_n.sum() / len(embeddings1_normalized)

def get_embeddings(model : torch.nn.Module, dataloader : torch.utils.data.DataLoader, device : torch.device) -> dict:
    """
    Evaluates a model on a given dataset and saves the embedding dictionary.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The dataloader for the dataset.

    Returns:
        dict: A dictionary containing the embedding arrays.
            Key 0: Embedding for anchor data points.
            Key 1: Embedding for positive data points.
    """

    emb1_list = []
    emb2_list = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            emb_anchor = model(data[0].to(device)).detach().cpu().numpy()
            emb_positive = model(data[1].to(device)).detach().cpu().numpy()
            for i in range(emb_anchor.shape[0]):
                emb1_list.append(emb_anchor[i])
                emb2_list.append(emb_positive[i])
            del emb_anchor, emb_positive
            torch.cuda.empty_cache()

    return {0: np.array(emb1_list), 1: np.array(emb2_list)}
