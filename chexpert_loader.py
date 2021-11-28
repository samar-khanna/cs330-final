import os
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import dataset, sampler, dataloader

from chexpert_utils import get_uncertain_cleaner, get_target_sampler


class ChexpertDataset(dataset.Dataset):
    chexpert_targets = ['No Finding',
                        'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                        'Support Devices']

    def __init__(self, data_path, path_to_csv, num_support, num_query, num_new_targets,
                 uncertain_cleaner, target_sampler, test_targets=None, im_size=(128, 128)):
        self.data_path = data_path
        self.df = pd.read_csv(path_to_csv)
        self.df['Path'] = self.df['Path'].apply(lambda p: p.replace('CheXpert-v1.0-small/', ''))

        self.num_new_targets = num_new_targets
        self.test_targets = test_targets
        if test_targets is not None:
            self.num_new_targets = len(test_targets)

        self.num_support = num_support
        self.num_query = num_query
        self.im_size = im_size
        self.target_sampler = target_sampler
        self.uncertain_cleaner = uncertain_cleaner

    def __len__(self):
        return len(self.df)

    def __getitem__(self, class_idxs):
        """
        Get task given subset of indices
        :param class_idxs:
        :return:
        """
        U, K = self.num_new_targets, len(class_idxs) - self.num_new_targets
        unk_class_idxs = self.test_targets if self.test_targets is not None else \
            np.random.choice(class_idxs, size=U, replace=False).astype(np.int)  # (U,)
        known_class_idxs = np.array([i for i in class_idxs if i not in unk_class_idxs], dtype=np.int)  # (K,)

        # Bool mask indicating non-nan rows for at least one of the classes in unk_class
        class_valid_mask = np.zeros(len(self.df), dtype=np.bool)
        for c in unk_class_idxs:
            class_col = self.df[self.chexpert_targets[c]].values
            class_valid_mask = class_valid_mask | ~np.isnan(class_col)

        # Valid image paths corresponding to at least one of the classes in unk_class_idxs
        valid_paths = self.df['Path'][class_valid_mask].values
        chexpert_classes = [self.chexpert_targets[c] for c in np.concatenate((known_class_idxs, unk_class_idxs))]
        valid_labels = self.df[chexpert_classes][class_valid_mask].values  # (N, K + U)

        # Replace uncertain labels (i.e. those with -1)
        valid_labels = self.uncertain_cleaner.clean(valid_labels)  # (N, K + U)

        # Sample num_support + num_query rows (i.e. image-label pairs) according to sampling strategy
        inds = self.target_sampler.sample(valid_labels, self.num_support + self.num_query)
        support_inds = inds[:self.num_support]
        query_inds = inds[self.num_support:]

        im_paths_support = valid_paths[support_inds]
        images_support = [self.load_image(os.path.join(self.data_path, p))
                          for p in im_paths_support]
        images_support = np.array(images_support, dtype=np.float32)  # (n_s, 1, h, w)
        labels_support = np.array(valid_labels[support_inds])  # (n_s, K + U)
        valid_support = ~np.isnan(labels_support)  # (n_s, K + U)
        known_support = np.concatenate(
            (valid_support[:, :K], np.zeros((valid_support.shape[0], U), dtype=np.bool)), axis=1
        )
        unknown_support = np.concatenate(
            (np.zeros((valid_support.shape[0], K), dtype=np.bool), valid_support[:, K:]), axis=1
        )

        im_paths_query = valid_paths[query_inds]
        images_query = [self.load_image(os.path.join(self.data_path, p))
                        for p in im_paths_query]
        images_query = np.array(images_query, dtype=np.float32)  # (n_q, 1, h, w)
        labels_query = np.array(valid_labels[query_inds])  # (n_q, K + U)
        valid_query = ~np.isnan(labels_query)  # (n_q, K + U)
        known_query = np.concatenate(
            (valid_query[:, :K], np.zeros((valid_query.shape[0], U), dtype=np.bool)), axis=1
        )
        unknown_query = np.concatenate(
            (np.zeros((valid_query.shape[0], K), dtype=np.bool), valid_query[:, K:]), axis=1
        )

        return torch.from_numpy(images_support), torch.from_numpy(labels_support), \
               torch.from_numpy(known_support), torch.from_numpy(unknown_support), \
               torch.from_numpy(images_query), torch.from_numpy(labels_query), \
               torch.from_numpy(known_query), torch.from_numpy(unknown_query)

    def load_image(self, im_path):
        # TODO: Should we normalize??
        im = Image.open(im_path)
        im = im.resize(self.im_size, Image.BILINEAR)
        return np.array(im)[np.newaxis, ...]/255


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
        total_targets_per_task,
        unk_targets_per_task,
        num_support,
        num_query,
        num_tasks_per_train_epoch,
        num_tasks_per_test_epoch,
        uncertain_strategy='positive',
        target_sampler_strategy='at_least_k',
        test_classes=None
):
    """
    Returns the train/val/test dataloaders for Chexpert
    :param data_path: Path to source directory with datasets
    :param batch_size: Size of each mini-batch
    :param total_targets_per_task: Number of known and novel classes to learn per task
    :param unk_targets_per_task: Number of novel classes to learn per task
    :param num_support: Number of instances in support dataset
    :param num_query: Number of instances in query dataset
    :param num_tasks_per_train_epoch: Number of tasks to sample per training epoch
    :param num_tasks_per_test_epoch: Number of tasks to sample per testing epoch
    :param uncertain_strategy: String specifying how to replace -1. uncertain labels in dataset
    :param target_sampler_strategy: Specify how to sample valid instances for each task/set of diseases
    :param test_classes: Hard set the indices of the test diseases
    :return: train/val/test dataloaders
    """
    uncertain_cleaner = get_uncertain_cleaner(uncertain_strategy)
    target_sampler = get_target_sampler(target_sampler_strategy)
    train_dataset = ChexpertDataset(data_path,
                                    os.path.join(data_path, 'train.csv'),
                                    num_support, 
                                    num_query,
                                    unk_targets_per_task,
                                    uncertain_cleaner=uncertain_cleaner,
                                    target_sampler=target_sampler)
    # TODO: Add validation dataset

    test_dataset = ChexpertDataset(data_path,
                                   os.path.join(data_path, 'valid.csv'),
                                   num_support, 
                                   num_query,
                                   unk_targets_per_task,
                                   test_targets=test_classes,
                                   uncertain_cleaner=uncertain_cleaner,
                                   target_sampler=target_sampler)

    classes = ChexpertDataset.chexpert_targets
    if test_classes is not None:
        test_idxs = test_classes
    else:
        test_idxs = np.random.choice(len(classes), unk_targets_per_task, replace=False)

    train_idxs = [i for i in range(len(classes)) if i not in test_idxs]

    train_loader = dataloader.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=ChexpertSampler(train_idxs, total_targets_per_task, num_tasks_per_train_epoch),
        num_workers=2,
        collate_fn=identity,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    # TODO: Fix unk_targets_per_task to be able to know known diseases during test time
    # TODO: Make work for different numbers of unk targets and total targets
    test_loader = dataloader.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        # Note: we use train_idxs coz test dataset already has hard-set test classes which will always
        # form the U unknown targets.
        sampler=ChexpertSampler(train_idxs, total_targets_per_task-unk_targets_per_task, num_tasks_per_test_epoch),
        num_workers=2,
        collate_fn=identity,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    return train_loader, test_loader, test_idxs

