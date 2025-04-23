import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

def subsample_instances(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices

class MergedDataset(Dataset):
    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """
    def __init__(self, labelled_dataset, unlabelled_dataset):
        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None

        # Create a dictionary to map id to index
        self.id_to_index = {}
        for i in tqdm(range(len(self))):
            _, _, uq_idx, _ = self.__getitem__(i)
            self.id_to_index[uq_idx] = i

    def __getitem__(self, item):
        if item < len(self.labelled_dataset):
            img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1
        else:
            img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0
        return img, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)

    def get_data_by_id(self, id):
        # Use the dictionary to get the index of the data
        index = self.id_to_index.get(id)
        if index is not None:
            return self.__getitem__(index)
        else:
            return None
    
        
class AffMergedDataset(Dataset):
    def __init__(self, merged_dataset, aff_dict):
        self.merged_dataset = merged_dataset
        self.labelled_dataset = merged_dataset.labelled_dataset
        self.unlabelled_dataset = merged_dataset.unlabelled_dataset
        self.aff_dict = aff_dict
        self.target_transform = None
    
    def __getitem__(self, item):
        img, label, uq_idx, labeled_or_not = self.merged_dataset[item]
        aff_id = self.aff_dict[uq_idx]
        aff_img = self.merged_dataset.get_data_by_id(aff_id)[0]

        return img, label, uq_idx, labeled_or_not, aff_img

    def __len__(self):
        return len(self.merged_dataset)
