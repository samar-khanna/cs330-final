"""Implementation of model-agnostic meta-learning for Omniglot."""

import argparse
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import autograd  # pylint: disable=unused-import
from torch.utils import tensorboard

import trainer_utils
import chexpert_loader
from model import conv_model

NUM_INPUT_CHANNELS = 1
NUM_HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
LOG_INTERVAL = 10
VAL_INTERVAL = LOG_INTERVAL * 5


class MAML:
    """Trains and assesses a MAML."""

    def __init__(
            self,
            num_outputs,
            num_inner_steps,
            inner_lr,
            learn_inner_lrs,
            outer_lr,
            log_dir
    ):
        """
        Inits MAML (vanilla) for multi-label few-shot novel disease detection
        :param num_outputs: Number of output channels in final layer of network (eqv to #novel classes)
        :param num_inner_steps: Number of inner-loop optimization steps
        :param inner_lr: learning rate for inner-loop optimization
        If learn_inner_lrs=True, inner_lr serves as the initialization of the learning rates.
        :param learn_inner_lrs: whether to learn the above
        :param outer_lr: learning rate for outer-loop optimization
        :param log_dir: path to logging directory
        """
        # TODO: Make genralisable to new models
        params, forward_fn = conv_model(
            NUM_CONV_LAYERS, NUM_INPUT_CHANNELS, NUM_HIDDEN_CHANNELS,
            num_outputs, KERNEL_SIZE, DEVICE
        )

        self._forward_func = forward_fn
        self._meta_parameters = params
        self._num_inner_steps = num_inner_steps
        self._inner_lrs = {
            k: torch.tensor(inner_lr, requires_grad=learn_inner_lrs)
            for k in self._meta_parameters.keys()
        }
        self._outer_lr = outer_lr

        self._optimizer = torch.optim.Adam(
            list(self._meta_parameters.values()) +
            list(self._inner_lrs.values()),
            lr=self._outer_lr
        )
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0

    def _forward(self, images, parameters):
        """
        Computes predicted classification logits.
        :param [Tensor] images: (N, c, h, w) batch of Chexpert images
        :param [dict[str, Tensor]] parameters: parameters to use for the computation
        :return: (N, num_targets) Tensor consisting of batch of logits
        """
        return self._forward_func(images, parameters)

    def _loss(self, pred, gt, keep_mask):
        """
        Computes masked binary cross entropy loss, ignoring the loss on invalid labels
        :param [Tensor] pred: (N, n_t) model predicted logits
        :param [Tensor] gt: (N, n_t) Ground truth labels
        :param [Tensor] keep_mask: (N, n_t) Boolean mask of which ground truth labels are valid for loss
        :return: Scalar mean loss value
        """
        gt = gt.clone()
        gt[~keep_mask] = 0.  # Arbitrary since these are masked out anyhoo
        unreduced = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
        loss = unreduced[keep_mask]
        return loss.mean()

    def _inner_loop(self, parameters, images, labels, keep_mask, train):   # pylint: disable=unused-argument
        """
        Computes adapted network parameters via MAML inner loop
        :param parameters: {param_name -> Tensor} Model parameters to update in inner loop
        :param images: (N_s, c, h, w) Task support set inputs
        :param labels: (N_s, n_t) Task support set labels (i.e. targets)
        :param keep_mask: (N_s, n_t) Boolean mask of valid labels for which to compute loss
        :param train: Whether we are training or evaluating
        :return:
            parameters (dict[str, Tensor]): adapted network parameters
            accuracies (list[float]): support set accuracy over the course of
                the inner loop, length num_inner_steps + 1
        """
        accuracies = []

        for i in range(self._num_inner_steps):
            out = self._forward(images, parameters)  # (N_s, U)
            loss = self._loss(out, labels, keep_mask)

            grad = autograd.grad(loss, parameters.values(), create_graph=train)
            for (k, p), g in zip(parameters.items(), grad):
                parameters[k] = p - self._inner_lrs[k]*g

            score = trainer_utils.score(out, labels, keep_mask)
            accuracies.append(score)

        # Calculate final post adaptation score
        out = self._forward(images, parameters)
        score = trainer_utils.score(out, labels, keep_mask)
        accuracies.append(score)

        return parameters, accuracies

    def _outer_step(self, task_batch, train):  # pylint: disable=unused-argument
        """
        Computer MAML loss and metrics on batch of tasks
        :param [Tuple] task_batch: Batch of tasks (each task is subset of diseases) from Chexpert DataLoader
        :param [bool] train: Whether we are training or evaluating
        :return:
            outer_loss (Tensor): mean MAML loss over the batch, scalar
            accuracies_support (ndarray): support set accuracy over the course of the inner loop,
                averaged over the task batch shape (num_inner_steps + 1,)
            accuracy_query (float): query set accuracy of the adapted parameters, averaged over the task batch
        """
        outer_loss_batch = []
        accuracies_support_batch = []
        accuracy_query_batch = []
        for task in task_batch:
            images_support, labels_support, known_support, unknown_support,\
            images_query, labels_query, known_query, unknown_query = task
            images_support = images_support.to(DEVICE)
            labels_support = labels_support.to(DEVICE)
            unknown_support = unknown_support.to(DEVICE)

            images_query = images_query.to(DEVICE)
            labels_query = labels_query.to(DEVICE)
            unknown_query = unknown_query.to(DEVICE)

            # self._optimizer.zero_grad()

            parameters = {k: torch.clone(v) for k, v in self._meta_parameters.items()}
            phi, accuracies = self._inner_loop(parameters, images_support, labels_support, unknown_support, train)

            out = self._forward(images_query, phi)  # (N*Kq, N)
            loss = self._loss(out, labels_query, unknown_query)

            query_acc = trainer_utils.score(out, labels_query, unknown_query)

            outer_loss_batch.append(loss)
            accuracies_support_batch.append(accuracies)
            accuracy_query_batch.append(query_acc)

        outer_loss = torch.mean(torch.stack(outer_loss_batch))
        accuracies_support = np.mean(
            accuracies_support_batch,
            axis=0
        )
        accuracy_query = np.mean(accuracy_query_batch)
        return outer_loss, accuracies_support, accuracy_query

    def train(self, dataloader_train, dataloader_val, writer):
        """
        Train MAML.
        Consmes dataloader_train to optimize MAML meta-parameters
        while periodically validating on dataloader_val, logging metrics, and
        saving checkpoints.
        :param [Dataloader] dataloader_train: Loader for training tasks
        :param [Dataloader] dataloader_val: Loader for validation tasks
        :param [SummaryWriter] writer: Tensorboard logger
        :return:
        """
        print(f'Starting training at iteration {self._start_train_step}.')
        for i_step, task_batch in enumerate(
                dataloader_train,
                start=self._start_train_step
        ):
            self._optimizer.zero_grad()
            outer_loss, accuracies_support, accuracy_query = (
                self._outer_step(task_batch, train=True)
            )
            outer_loss.backward()
            self._optimizer.step()

            if i_step % LOG_INTERVAL == 0:
                print(
                    f'Iteration {i_step}: '
                    f'loss: {outer_loss.item():.3f}, '
                    f'pre-adaptation support accuracy: '
                    f'{accuracies_support[0]:.3f}, '
                    f'post-adaptation support accuracy: '
                    f'{accuracies_support[-1]:.3f}, '
                    f'post-adaptation query accuracy: '
                    f'{accuracy_query:.3f}'
                )
                writer.add_scalar('loss/train', outer_loss.item(), i_step)
                writer.add_scalar(
                    'train_accuracy/pre_adapt_support',
                    accuracies_support[0],
                    i_step
                )
                writer.add_scalar(
                    'train_accuracy/post_adapt_support',
                    accuracies_support[-1],
                    i_step
                )
                writer.add_scalar(
                    'train_accuracy/post_adapt_query',
                    accuracy_query,
                    i_step
                )

            if i_step % VAL_INTERVAL == 0:
                losses = []
                accuracies_pre_adapt_support = []
                accuracies_post_adapt_support = []
                accuracies_post_adapt_query = []
                for val_task_batch in dataloader_val:
                    outer_loss, accuracies_support, accuracy_query = (
                        self._outer_step(val_task_batch, train=False)
                    )
                    losses.append(outer_loss.item())
                    accuracies_pre_adapt_support.append(accuracies_support[0])
                    accuracies_post_adapt_support.append(accuracies_support[-1])
                    accuracies_post_adapt_query.append(accuracy_query)
                loss = np.mean(losses)
                accuracy_pre_adapt_support = np.mean(
                    accuracies_pre_adapt_support
                )
                accuracy_post_adapt_support = np.mean(
                    accuracies_post_adapt_support
                )
                accuracy_post_adapt_query = np.mean(
                    accuracies_post_adapt_query
                )
                print(
                    f'Validation: '
                    f'loss: {loss:.3f}, '
                    f'pre-adaptation support accuracy: '
                    f'{accuracy_pre_adapt_support:.3f}, '
                    f'post-adaptation support accuracy: '
                    f'{accuracy_post_adapt_support:.3f}, '
                    f'post-adaptation query accuracy: '
                    f'{accuracy_post_adapt_query:.3f}'
                )
                writer.add_scalar('loss/val', loss, i_step)
                writer.add_scalar(
                    'val_accuracy/pre_adapt_support',
                    accuracy_pre_adapt_support,
                    i_step
                )
                writer.add_scalar(
                    'val_accuracy/post_adapt_support',
                    accuracy_post_adapt_support,
                    i_step
                )
                writer.add_scalar(
                    'val_accuracy/post_adapt_query',
                    accuracy_post_adapt_query,
                    i_step
                )

            if i_step % SAVE_INTERVAL == 0:
                self._save(i_step)

    def test(self, dataloader_test, num_tasks_per_epoch):
        """
        Evaluate MAML on test tasks
        :param [Dataloader] dataloader_test: Loader for test tasks (each task is subset of diseases)
        :param [int] num_tasks_per_epoch: Number of test tasks
        :return:
        """
        accuracies = []
        for task_batch in dataloader_test:
            _, _, accuracy_query = self._outer_step(task_batch, train=False)
            accuracies.append(accuracy_query)
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(num_tasks_per_epoch)
        print(
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )

    def load(self, checkpoint_step):
        """
        Loads model checkpoint parameters
        :param [int] checkpoint_step: iteration of checkpoint to load
        :raises ValueError: if checkpoint for checkpoint_step is not found
        """

        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{checkpoint_step}.pt'
        )
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self._meta_parameters = state['meta_parameters']
            self._inner_lrs = state['inner_lrs']
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
            self._start_train_step = checkpoint_step + 1
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    def _save(self, checkpoint_step):
        """
        Saves parameters and optimizer state_dictionary as a checkpoint
        :param [int] checkpoint_step: Iteration with which to label checkpoint
        :return:
        """

        optimizer_state_dict = self._optimizer.state_dict()
        torch.save(
            dict(meta_parameters=self._meta_parameters,
                 inner_lrs=self._inner_lrs,
                 optimizer_state_dict=optimizer_state_dict),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')


def main(args):
    log_dir = args.log_dir
    if log_dir is None:
        test_class_str = '-'.join([str(i) for i in args.test_classes]) \
            if args.test_classes is not None else 'random'
        log_dir = f'./logs/maml/chexpert.tasks:{args.num_tasks}.test:{test_class_str}' \
                  f'.new_targets:{args.num_targets}' \
                  f'.support:{args.num_support}.query:{args.num_query}' \
                  f'.inner_steps:{args.num_inner_steps}.inner_lr:{args.inner_lr}' \
                  f'.learn_inner_lrs:{args.learn_inner_lrs}.outer_lr:{args.outer_lr}' \
                  f'.uncertain:{args.uncertain_cleaner}.target_sampler:{args.target_sampler}' \
                  f'.batch_size:{args.batch_size}'
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    maml = MAML(
        args.total_targets,
        args.num_inner_steps,
        args.inner_lr,
        args.learn_inner_lrs,
        args.outer_lr,
        log_dir
    )

    if args.checkpoint_step > -1:
        maml.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    num_training_tasks = args.batch_size * (args.num_tasks - args.checkpoint_step - 1)
    num_testing_tasks = args.batch_size * 8
    train_loader, test_loader, test_idxs = chexpert_loader.get_chexpert_dataloader(
        args.data_path, args.batch_size, args.total_targets, args.num_targets,
        args.num_support, args.num_query, num_training_tasks, num_testing_tasks,
        uncertain_strategy=args.uncertain_cleaner, target_sampler_strategy=args.target_sampler,
        test_classes=args.test_classes
    )

    diseases = chexpert_loader.ChexpertDataset.chexpert_targets
    if not args.test:
        print(
            f'Num test classes {args.num_targets}: {[diseases[i] for i in test_idxs]}\n'
            f'Training on {args.num_tasks} tasks per epoch with composition: '
            f'num_targets_per_task={args.num_targets}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )

        maml.train(train_loader, test_loader, writer)
    else:
        print(
            f'Num test classes {args.num_targets}: {[diseases[i] for i in test_idxs]}'
            f'num_targets_per_task={args.num_targets}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        maml.test(test_loader, num_testing_tasks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a MAML!')
    parser.add_argument('--data_path', type=str, default=None,
                        help='directory containing all data')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')

    parser.add_argument('--num_support', type=int, default=10,
                        help='Total number of samples for task support dataset')
    parser.add_argument('--num_query', type=int, default=16,
                        help='Total number of samples for task support dataset')
    parser.add_argument('--num_tasks', type=int, default=4000,
                        help='Total number of tasks to sample for training (eqv: num_train_iter)')
    parser.add_argument('--total_targets', type=int, default=8,
                        help='Total number of known and unknown diseases to sample for each task')
    parser.add_argument('--num_targets', type=int, default=4,
                        help='Number of new/unknown diseases to sample for each task')
    # parser.add_argument('--num_train_iterations', type=int, default=4000,
    #                     help='number of outer-loop updates to train for')

    parser.add_argument('--test_classes', default=None, type=int, nargs='+',
                        help='Specify indices of custom test classes to hold out')
    parser.add_argument('--uncertain_cleaner', default='positive', type=str,
                        help='Specify how to replace uncertain targets in dataset')
    parser.add_argument('--target_sampler', default='at_least_k', type=str,
                        help='Specify how to sample targets for a task')

    parser.add_argument('--num_inner_steps', type=int, default=1,
                        help='number of inner-loop updates')
    parser.add_argument('--inner_lr', type=float, default=0.1,
                        help='inner-loop learning rate initialization')
    parser.add_argument('--learn_inner_lrs', default=False, action='store_true',
                        help='whether to optimize inner-loop learning rates')
    parser.add_argument('--outer_lr', type=float, default=0.001,
                        help='outer-loop learning rate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of tasks per outer-loop update')

    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))

    main_args = parser.parse_args()
    main(main_args)
