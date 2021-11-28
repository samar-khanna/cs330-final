import numpy as np

from typing import Type


class TargetSampler:
    def __init__(self, **kwargs):
        pass

    def sample(self, labels, num_samples, unk_idxs, **kwargs):
        raise NotImplementedError("Abstract base class. Should be subclassed")


class RandomTargetSampler(TargetSampler):
    def __init__(self, **kwargs):
        super().__init__()

    def sample(self, labels, num_samples, unk_idxs, **kwargs):
        """
        Randomly samples num_samples rows from labels
        :param labels: (N, num_targets) array of values (either 1, 0, or nan)
        :param num_samples: total number of samples
        :return: (num_samples,) indices in labels that form the task
        """
        inds = np.random.choice(len(labels), size=num_samples, replace=False)
        return inds


class KTargetSampler(TargetSampler):
    def __init__(self, k=2):
        super().__init__()
        self.k = k

    def sample(self, labels, num_samples, unk_idxs, **kwargs):
        """
        Samples rows that have at least k non-empty values for the target classes
        :param labels: (N, num_targets) array of values (either 1, 0, or nan)
        :param num_samples: total number of samples
        :return: (num_samples,) indices in labels that form the task
        """
        valid = np.sum(~np.isnan(labels), axis=1) >= self.k
        possible, = np.nonzero(valid)
        if len(possible) >= num_samples:
            inds = np.random.choice(possible, size=num_samples, replace=False)
        else:
            print(f"Warning: Fewer than {num_samples} options. Defaulting to random sampling")
            inds = RandomTargetSampler().sample(labels, num_samples, unk_idxs)
        return inds


class KnownUnknownTargetSampler(KTargetSampler):
    def __init__(self, k=2):
        super().__init__(k)

    def sample(self, labels, num_samples, unk_idxs, **kwargs):
        """
        Samples rows that have at least k non-empty values for both known and unknown classes
        :param labels: (N, num_targets) array of values (either 1, 0, or nan)
        :param num_samples: total number of samples
        :param unk_idxs: List of indices of unknown class columns
        :return: (num_samples,) indices in labels that form the task
        """
        known_idxs = [i for i in range(labels.shape[1]) if i not in unk_idxs]
        known_valid = np.sum(~np.isnan(labels[:, known_idxs])) >= self.k
        unknown_valid = np.sum(~np.isnan(labels[:, unk_idxs])) >= self.k
        possible, = np.nonzero(known_valid & unknown_valid)
        if len(possible) >= num_samples:
            inds = np.random.choice(possible, size=num_samples, replace=False)
        else:
            print(f"Warning: Fewer than {num_samples} options. Defaulting to k-target sampling")
            inds = super().sample(labels, num_samples, unk_idxs)
        return inds


class TestTargetSampler(TargetSampler):
    def __init__(self):
        super().__init__()
        self.count = 0

    def sample(self, labels, num_samples, **kwargs):
        if self.count + num_samples > labels.shape[0]:
            self.count = 0
        inds = np.arange(self.count, self.count + num_samples, dtype=np.int)
        self.count += num_samples
        return inds


def get_target_sampler(strategy: str, **kwargs) -> TargetSampler:
    if 'random' in strategy.lower():
        return RandomTargetSampler()
    elif 'at_least_k' in strategy.lower():
        return KTargetSampler(**kwargs)
    elif 'known_unknown' in strategy.lower():
        return KnownUnknownTargetSampler(**kwargs)
    elif 'test' in strategy.lower():
        return TestTargetSampler()
    else:
        raise NotImplementedError(f"Invalid target sampling strategy: {strategy}")


class UncertainCleaner:
    def __init__(self):
        pass

    def clean(self, labels):
        raise NotImplementedError("Abstract base class. Should be subclassed")


class ReplaceWithPositive(UncertainCleaner):
    def __init__(self):
        super().__init__()

    def clean(self, labels):
        """
        Replaces all uncertain labels with positive labels
        :param labels: (N, num_targets) array of values (either 1, 0, -1, or nan)
        :return: (N, num_targets) array of values (either 1, 0, or nan)
        """
        labels[labels == -1] = 1
        return labels


def get_uncertain_cleaner(strategy) -> UncertainCleaner:
    if 'positive' in strategy.lower():
        return ReplaceWithPositive()
    else:
        raise NotImplementedError(f"Invalid uncertain cleaning strategy: {strategy}")
