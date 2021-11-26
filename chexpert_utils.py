import numpy as np

from typing import Type


class TargetSampler:
    def __init__(self, **kwargs):
        pass

    def sample(self, labels, num_samples):
        raise NotImplementedError("Abstract base class. Should be subclassed")


class RandomTargetSampler(TargetSampler):
    def __init__(self, **kwargs):
        super().__init__()

    def sample(self, labels, num_samples):
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

    def sample(self, labels, num_samples):
        """
        Samples rows that have at least k non-empty values for the target classes
        :param labels: (N, num_targets) array of values (either 1, 0, or nan)
        :param num_samples: total number of samples
        :return: (num_samples,) indices in labels that form the task
        """
        valid = np.sum(~np.isnan(labels), axis=1) >= self.k
        possible = np.nonzero(valid)
        if len(possible) >= num_samples:
            inds = np.random.choice(possible, size=num_samples, replace=False)
        else:
            print(f"Warning: Fewer than {num_samples} options. Defaulting to random sampling")
            inds = RandomTargetSampler().sample(labels, num_samples)
        return inds


def get_target_sampler(strategy: str) -> Type[TargetSampler]:
    if 'random' in strategy.lower():
        return RandomTargetSampler
    elif 'at_least_k' in strategy.lower():
        return KTargetSampler
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


def get_uncertain_cleaner(strategy) -> Type[UncertainCleaner]:
    if 'positive' in strategy.lower():
        return ReplaceWithPositive
    else:
        raise NotImplementedError(f"Invalid uncertain cleaning strategy: {strategy}")
