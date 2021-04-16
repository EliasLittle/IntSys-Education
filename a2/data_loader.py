import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


class SimpleDataset(Dataset):
    """SimpleDataset [summary]

    [extended_summary]

    :param path_to_csv: [description]
    :type path_to_csv: [type]
    """
    def __init__(self, path_to_csv, transform=None):
        self.data = np.genfromtxt(path_to_csv, delimiter=",")

        self.transform = transform

    def __len__(self):
        """__len__ returns length of dataset"""
        return len(self.data)

    def __getitem__(self, index):
        """__getitem__ [summary]

        [extended_summary]

        :param index: [description]
        :type index: [type]
        """
        ## This returns only ONE sample from the dataset, for a given index.
        ## The returned sample should be a tuple (x, y) where x is your input
        ## vector and y is your label
        ## Before returning your sample, you should check if there is a transform
        ## sepcified, and apply that transform to your sample
        # Eg:
        # if self.transform:
        #   sample = self.transform(sample)
        ## Remember to convert the x and y into torch tensors.
        x, y = torch.from_numpy(self.data[:,:-1]).float(), torch.from_numpy(self.data[:,-1]).float()
        sample = x[index], y[ index]
        if self.transform is None:
            return sample
        else:
            return self.transform(sample)


def get_data_loaders(path_to_csv,
                     transform_fn=None,
                     train_val_test=[0.8, 0.2, 0.2],
                     batch_size=32):
    """get_data_loaders [summary]

    [extended_summary]

    :param path_to_csv: [description]
    :type path_to_csv: [type]
    :param train_val_test: [description], defaults to [0.8, 0.2, 0.2]
    :type train_val_test: list, optional
    :param batch_size: [description], defaults to 32
    :type batch_size: int, optional
    :return: [description]
    :rtype: [type]
    """
    # First we create the dataset given the path to the .csv file
    dataset = SimpleDataset(path_to_csv, transform=transform_fn)

    # Then, we create a list of indices for all samples in the dataset.
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    ## TODO: Rewrite this section so that the indices for each dataset split
    ## are formed.

    ## BEGIN: YOUR CODE
    test_size = int(train_val_test[-1]*dataset_size)
    train_val_size = dataset_size - test_size
    train_size = int(train_val_test[0]*train_val_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:-1*test_size]
    test_indices = indices[-1*test_size:]
    ## END: YOUR CODE

    # Now, we define samplers for each of the train, val and test data
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    return train_loader, val_loader, test_loader
