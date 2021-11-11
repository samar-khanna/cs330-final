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
        im = Image.open(im_path)
        im = im.resize(self.im_size, Image.BILINEAR)
        return np.array(im)

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
    def __init__(self, split_idxs, num_way, num_tasks):
        """Inits OmniglotSampler.

        Args:
            split_idxs (range): indices that comprise the
                training/validation/test split
            num_way (int): number of classes per task
            num_tasks (int): number of tasks to sample
        """
        super().__init__(None)
        self._split_idxs = split_idxs
        self._num_way = num_way
        self._num_tasks = num_tasks

    def __iter__(self):
        return (
            np.random.default_rng().choice(
                self._split_idxs,
                size=self._num_way,
                replace=False
            ) for _ in range(self._num_tasks)
        )  # generator of len num_tasks, each elem is size N array of indices

    def __len__(self):
        return self._num_tasks






NUM_TRAIN_CLASSES = 1100
NUM_VAL_CLASSES = 100
NUM_TEST_CLASSES = 423
NUM_SAMPLES_PER_CLASS = 20


def load_image(file_path):
    """Loads and transforms an Omniglot image.

    Args:
        file_path (str): file path of image

    Returns:
        a Tensor containing image data
            shape (1, 28, 28)
    """
    x = imageio.imread(file_path)
    x = torch.tensor(x, dtype=torch.float32).reshape([1, 28, 28])
    x = x / 255.0
    return 1 - x


class OmniglotDataset(dataset.Dataset):
    """Omniglot dataset for meta-learning.

    Each element of the dataset is a task. A task is specified with a key,
    which is a tuple of class indices (no particular order). The corresponding
    value is the instantiated task, which consists of sampled (image, label)
    pairs.
    """

    _BASE_PATH = './omniglot_resized'
    _GDD_FILE_ID = '1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI'

    def __init__(self, num_support, num_query):
        """Inits OmniglotDataset.

        Args:
            num_support (int): number of support examples per class
            num_query (int): number of query examples per class
        """
        super().__init__()


        # if necessary, download the Omniglot dataset
        if not os.path.isdir(self._BASE_PATH):
            gdd.GoogleDriveDownloader.download_file_from_google_drive(
                file_id=self._GDD_FILE_ID,
                dest_path=f'{self._BASE_PATH}.zip',
                unzip=True
            )

        # get all character folders
        self._character_folders = glob.glob(
            os.path.join(self._BASE_PATH, '*/*/'))
        assert len(self._character_folders) == (
            NUM_TRAIN_CLASSES + NUM_VAL_CLASSES + NUM_TEST_CLASSES
        )

        # shuffle characters
        np.random.default_rng(0).shuffle(self._character_folders)

        # check problem arguments
        assert num_support + num_query <= NUM_SAMPLES_PER_CLASS
        self._num_support = num_support
        self._num_query = num_query

    def __getitem__(self, class_idxs):
        """Constructs a task.

        Data for each class is sampled uniformly at random without replacement.
        The ordering of the labels corresponds to that of class_idxs.

        Args:
            class_idxs (tuple[int]): class indices that comprise the task

        Returns:
            images_support (Tensor): task support images
                shape (num_way * num_support, channels, height, width)
            labels_support (Tensor): task support labels
                shape (num_way * num_support,)
            images_query (Tensor): task query images
                shape (num_way * num_query, channels, height, width)
            labels_query (Tensor): task query labels
                shape (num_way * num_query,)
        """
        images_support, images_query = [], []
        labels_support, labels_query = [], []

        for label, class_idx in enumerate(class_idxs):
            # get a class's examples and sample from them
            all_file_paths = glob.glob(
                os.path.join(self._character_folders[class_idx], '*.png')
            )
            sampled_file_paths = np.random.default_rng().choice(
                all_file_paths,
                size=self._num_support + self._num_query,
                replace=False
            )
            images = [load_image(file_path) for file_path in sampled_file_paths]

            # split sampled examples into support and query
            images_support.extend(images[:self._num_support])
            images_query.extend(images[self._num_support:])
            labels_support.extend([label] * self._num_support)
            labels_query.extend([label] * self._num_query)

        # aggregate into tensors
        images_support = torch.stack(images_support)  # shape (N*S, C, H, W)
        labels_support = torch.tensor(labels_support)  # shape (N*S)
        images_query = torch.stack(images_query)
        labels_query = torch.tensor(labels_query)

        return images_support, labels_support, images_query, labels_query


class OmniglotSampler(sampler.Sampler):
    """Samples task specification keys for an OmniglotDataset."""

    def __init__(self, split_idxs, num_way, num_tasks):
        """Inits OmniglotSampler.

        Args:
            split_idxs (range): indices that comprise the
                training/validation/test split
            num_way (int): number of classes per task
            num_tasks (int): number of tasks to sample
        """
        super().__init__(None)
        self._split_idxs = split_idxs
        self._num_way = num_way
        self._num_tasks = num_tasks

    def __iter__(self):
        return (
            np.random.default_rng().choice(
                self._split_idxs,
                size=self._num_way,
                replace=False
            ) for _ in range(self._num_tasks)
        )

    def __len__(self):
        return self._num_tasks


def identity(x):
    return x


def get_omniglot_dataloader(
        split,
        batch_size,
        num_way,
        num_support,
        num_query,
        num_tasks_per_epoch
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

    if split == 'train':
        split_idxs = range(NUM_TRAIN_CLASSES)
    elif split == 'val':
        split_idxs = range(
            NUM_TRAIN_CLASSES,
            NUM_TRAIN_CLASSES + NUM_VAL_CLASSES
        )
    elif split == 'test':
        split_idxs = range(
            NUM_TRAIN_CLASSES + NUM_VAL_CLASSES,
            NUM_TRAIN_CLASSES + NUM_VAL_CLASSES + NUM_TEST_CLASSES
        )
    else:
        raise ValueError

    return dataloader.DataLoader(
        dataset=OmniglotDataset(num_support, num_query),
        batch_size=batch_size,
        sampler=OmniglotSampler(split_idxs, num_way, num_tasks_per_epoch),
        num_workers=2,
        collate_fn=identity,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

