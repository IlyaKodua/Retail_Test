import torch
import ruclip
from PIL import Image
import numpy as np


class ImageComparer:

    def __init__(self, device: torch.device) -> None:
        """
        Initializes the model and preprocessing pipeline.

        Args:
            device (torch.device): The device to use for the model.
        """
        model, preprocess = ruclip.load("ruclip-vit-base-patch16-384", device=device)
        self.device = device
        self.model = model.visual
        self.preprocess = preprocess.image_transform

    def __load_img(self, path):
        """
        Loads and preprocesses an image from the specified path.

        Args:
            path (str): Path to the image file.

        Returns:
            torch.Tensor: The preprocessed image as a tensor.
        """
        img = Image.open(path).convert("RGB")
        img = self.preprocess(img)
        return img

    def compare(self, path1: str, path2: str) -> float:
        """
        Compares two images using their embeddings and cosine similarity.

        Args:
            self (nn.Module): The model instance.
            path1 (str): Path to the first image.
            path2 (str): Path to the second image.

        Returns:
            float: Norm cosine similarity between the embeddings of the two images.
        """
        img1 = self.__load_img(path1).to(self.device).unsqueeze(0)
        img2 = self.__load_img(path2).to(self.device).unsqueeze(0)

        emb1 = self.model(img1).detach().cpu().numpy()[0]
        emb2 = self.model(img2).detach().cpu().numpy()[0]

        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        lvl = (1 + cos_sim) / 2

        return lvl
    

if __name__=='__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_comparer = ImageComparer(device)

    path1 = "dataset/30041783_1.jpg"
    path2 = "dataset/30041783_2.jpg"

    print(img_comparer.compare(path1, path2))


