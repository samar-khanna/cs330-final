import os
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import dataset, sampler, dataloader


class ChexpertDataset(dataset.Dataset):
    chexpert_targets = ['No Finding',
                        'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                        'Support Devices']

    def __init__(self, data_path, path_to_csv, num_support, num_query,
                 sampling_strategy='random', uncertain_strategy='positive', im_size=(320, 320)):
        self.data_path = data_path
        self.df = pd.read_csv(path_to_csv)
        self.num_support = num_support
        self.num_query = num_query
        self.im_size = im_size
        self.sampling_func = ChexpertDataset.get_sampling_func(sampling_strategy)
        self.uncertain_func = ChexpertDataset.get_uncertain_func(uncertain_strategy)

    @staticmethod
    def get_uncertain_func(strategy):
        if strategy == 'positive':
            return ChexpertDataset.replace_with_positive
        else:
            raise NotImplementedError(f'Invalid strategy {strategy}')

    @staticmethod
    def get_sampling_func(strategy):
        if strategy == 'random':
            return ChexpertDataset.random_sampling
        else:
            raise NotImplementedError(f'Invalid strategy {strategy}')

    def __getitem__(self, class_idxs):
        """
        Get task given subset of indices
        :param class_idxs:
        :return:
        """
        # TODO: Fix for alpha-beta
        known_class_idxs, unk_class_idxs = [], class_idxs

        # Bool mask indicating non-nan rows for at least one of the classes in unk_class
        class_valid_mask = np.zeros(len(self.df), dtype=np.bool)
        for c in unk_class_idxs:
            class_col = self.df[self.chexpert_targets[c]].values
            class_valid_mask = class_valid_mask | ~np.isnan(class_col)

        # Valid image paths corresponding to at least one of the classes in unk_class_idxs
        valid_paths = self.df['Path'][class_valid_mask]
        valid_labels = self.df[self.chexpert_targets[unk_class_idxs]][class_valid_mask].values

        # Replace uncertain labels
        valid_labels = self.uncertain_func(valid_labels)

        inds = self.sampling_func(valid_labels, self.num_support + self.num_query)
        support_inds = inds[:self.num_support]
        query_inds = inds[self.num_support:]

        im_paths_support = valid_paths[support_inds]
        images_support = [self.load_image(os.path.join(self.data_path, p))
                          for p in im_paths_support]
        images_support = np.array(images_support, dtype=np.float32)
        labels_support = np.array(valid_labels[support_inds])
        mask_support = np.isnan(labels_support)

        im_paths_query = valid_paths[query_inds]
        images_query = [self.load_image(os.path.join(self.data_path, p))
                          for p in im_paths_query]
        images_query = np.array(images_query, dtype=np.float32)
        labels_query = np.array(valid_labels[query_inds])
        mask_query = np.isnan(labels_query)

        return torch.from_numpy(images_support), torch.from_numpy(labels_support), torch.from_numpy(mask_support), \
            torch.from_numpy(images_query), torch.from_numpy(labels_query), torch.from_numpy(mask_query)

    def load_image(self, im_path):
        # TODO: Should we normalize??
        im = Image.open(im_path)
        im = im.resize(self.im_size, Image.BILINEAR)
        return np.array(im/255)

    @staticmethod
    def random_sampling(labels, num_samples):
        inds = np.random.choice(len(labels), size=num_samples, replace=False)
        return inds

    @staticmethod
    def replace_with_positive(labels):
        labels[labels == -1] = 1
        return labels





# TODO: Code from HW2
class ChexpertSampler(sampler.Sampler):
    def __init__(self, class_idxs, num_targets_per_task, num_tasks):
        """Inits OmniglotSampler.

        Args:
            class_idxs (int): indices of classes (targets)
            num_targets_per_task (range): number of targets/classes per task
            num_tasks (int): number of tasks to sample
        """
        super().__init__(None)
        self._class_idxs = class_idxs
        self._num_targets_per_task = num_targets_per_task
        self._num_tasks = num_tasks

    def __iter__(self):
        return (
            np.random.default_rng().choice(
                self._class_idxs,
                size=self._num_targets_per_task,
                replace=False
            ) for _ in range(self._num_tasks)
        )  # generator of len num_tasks, each elem is size N array of indices

    def __len__(self):
        return self._num_tasks

def identity(x):
    return x

def get_chexpert_dataloader(
        data_path,
        batch_size,
        num_test_class,
        num_targets_per_task,
        num_support,
        num_query,
        num_tasks_per_epoch,
        test_classes=None
):
    """Returns a dataloader.DataLoader for Omniglot.

    Args:
        split (str): one of 'train', 'val', 'test'
        batch_size (int): number of tasks per batch
        num_way (int): number of classes per task
        num_support (int): number of support examples per class
        num_query (int): number of query examples per class
        num_tasks_per_epoch (int): number of tasks before DataLoader is
            exhausted
    """

    train_dataset = OmniglotDataset(data_path,
                                    os.path.join(data_path, 'train.csv'),
                                    num_support, 
                                    num_query)
    # TODO: Add validation dataset

    test_dataset = OmniglotDataset(data_path,
                                   os.path.join(data_path, 'valid.csv'),
                                   num_support, 
                                   num_query)
    
    if test_classes:
        raise NotImplementedError
    else:
        classes = ChexpertDataset.chexpert_targets
        test_idxs = np.random.choice(len(classes), num_test_class, replace=False)
        train_idxs = [i for i in range(len(classes)) if i not in test_idxs]       

    train_dataloader = dataloader.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=ChexpertSampler(train_idxs, num_targets_per_task, num_tasks_per_epoch),
        num_workers=2,
        collate_fn=identity,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    test_dataloader = dataloader.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        sampler=ChexpertSampler(test_idxs, num_targets_per_task, num_tasks_per_epoch),
        num_workers=2,
        collate_fn=identity,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    return train_loader, test_loader

