from torch.utils.data import Dataset
import torch
import glob
from PIL import Image
from torchvision import transforms


class RetailDataset(Dataset):
    """
    Custom dataset class for retail image retrieval tasks.

    Args:
        root_dir (str): Path to the directory containing images.
        preprocess (callable): Preprocessing function to apply to images.

    Attributes:
        root_dir (str): Path to the root directory.
        files (list): List of image paths in the dataset.
        preprocess (callable): Preprocessing function.
        uniq_labels_list (list): List of unique labels (class names).
        len_classes (int): Number of unique classes.
    """

    def __init__(self, root_dir : str, preprocess : transforms) -> None:
        self.root_dir = root_dir
        self.files = sorted(glob.glob(root_dir + "/*.jpg", recursive=True))
        self.preprocess = preprocess

        labels_list = []
        for file in self.files:
            labels_list.append(file.split("/")[-1].split("_")[0])

        self.labels_list = self.__find_duplicates(labels_list)
        self.labels_list = sorted(list(self.labels_list))
        self.len_classes = len(self.labels_list)
        print("Number of classes:", self.len_classes)

    
    def __find_duplicates(self, lst : list) -> list:
        """
        Finds duplicate elements in a list.

        Args:
            lst (list): The list to search for duplicates.

        Returns:
            list: A list containing the duplicate elements.
        """

        count_dict = {}
        for num in lst:
            count_dict[num] = count_dict.get(num, 0) + 1

        duplicates = [key for key, value in count_dict.items() if value > 1]
        return duplicates

    def __len__(self):
        """
        Returns the length of the dataset (number of unique labels).
        """
        return len(self.labels_list)

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

    def __getitem__(self, idx):
        """
        Returns an anchor-positive image pair and corresponding label.

        Args:
            idx (int): Index of the image in the dataset.

        Returns:
            tuple: (anchor_img, positive_img, label)
        """

        anchor_path = self.root_dir + "/" + self.labels_list[idx] + "_1.jpg"
        positive_path = self.root_dir + "/" + self.labels_list[idx] + "_2.jpg"

        anchor_img = self.__load_img(anchor_path)
        positive_img = self.__load_img(positive_path)

        return anchor_img, positive_img, torch.tensor(idx)

