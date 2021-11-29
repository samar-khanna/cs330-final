import torch
import argparse
import numpy as np
from torch import autograd
from torch.utils import tensorboard

import trainer_utils
import chexpert_loader
from maml import MAML, DEVICE


class AlphaMAML(MAML):
    def __init__(self, num_outputs, num_inner_steps, inner_lr, alpha_inner_steps, alpha_inner_lr,
                 learn_inner_lrs, outer_lr, log_dir):
        """
        Inits an Alpha MAML
        :param num_outputs:
        :param num_inner_steps:
        :param inner_lr:
        :param learn_inner_lrs:
        :param outer_lr:
        :param log_dir:
        """
        super().__init__(num_outputs, num_inner_steps, inner_lr, learn_inner_lrs, outer_lr, log_dir)
        self._alpha_inner_steps = alpha_inner_steps
        self._alpha_inner_lrs = {
            k: torch.tensor(alpha_inner_lr, requires_grad=learn_inner_lrs)
            for k in self._meta_parameters.keys()
        }
        self._optimizer = torch.optim.Adam(
            list(self._meta_parameters.values()) +
            list(self._inner_lrs.values()) +
            list(self._alpha_inner_lrs.values()),
            lr=self._outer_lr
        )

    def _alpha_inner_loop(self, phi, images_query, labels_query, known_query, train):
        accuracies = []

        for i in range(self._num_inner_steps):
            out = self._forward(images_query, phi)  # (N_s, U)
            loss = self._loss(out, labels_query, known_query)

            grad = autograd.grad(loss, phi.values(), create_graph=train)
            for (k, p), g in zip(phi.items(), grad):
                phi[k] = p - self._alpha_inner_lrs[k]*g

            score = trainer_utils.score(out, labels_query, known_query)
            accuracies.append(score)

        # Calculate final post adaptation score
        out = self._forward(images_query, phi)
        score = trainer_utils.score(out, labels_query, known_query)
        accuracies.append(score)

        return phi, accuracies

    def _outer_step(self, task_batch, train):
        outer_loss_batch = []
        accuracies_support_batch = []
        accuracy_query_batch = []
        for task in task_batch:
            images_support, labels_support, known_support, unknown_support,\
            images_query, labels_query, known_query, unknown_query = task

            images_support = images_support.to(DEVICE)
            labels_support = labels_support.to(DEVICE)
            known_support = known_support.to(DEVICE)
            unknown_support = unknown_support.to(DEVICE)

            images_query = images_query.to(DEVICE)
            labels_query = labels_query.to(DEVICE)
            known_query = known_query.to(DEVICE)
            unknown_query = unknown_query.to(DEVICE)

            parameters = {k: torch.clone(v) for k, v in self._meta_parameters.items()}
            phi, accuracies = self._inner_loop(parameters, images_support, labels_support,
                                               known_support | unknown_support, train)

            phi, _ = self._alpha_inner_loop(phi, images_query, labels_query, known_query, train)

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


def main(args):
    log_dir = args.log_dir
    if log_dir is None:
        test_class_str = '-'.join([str(i) for i in args.test_classes]) \
            if args.test_classes is not None else 'random'
        log_dir = f'./logs/amaml/chexpert.tasks:{args.num_tasks}.test:{test_class_str}' \
                  f'.new_targets:{args.num_targets}.total_targets{args.total_targets}' \
                  f'.support:{args.num_support}.query:{args.num_query}' \
                  f'.inner_steps:{args.num_inner_steps}.inner_lr:{args.inner_lr}' \
                  f'.alpha_inner_steps:{args.alpha_inner_steps}.alpha_inner_lr:{args.alpha_inner_lr}' \
                  f'.learn_inner_lrs:{args.learn_inner_lrs}.outer_lr:{args.outer_lr}' \
                  f'.uncertain:{args.uncertain_cleaner}.target_sampler:{args.target_sampler}' \
                  f'.batch_size:{args.batch_size}'
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    maml = AlphaMAML(
        args.total_targets,
        args.num_inner_steps,
        args.inner_lr,
        args.alpha_inner_steps,
        args.alpha_inner_lr,
        args.learn_inner_lrs,
        args.outer_lr,
        log_dir
    )

    if args.checkpoint_step > -1:
        maml.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    target_sampler_strategy = 'at_least_k' if args.total_targets <= args.num_targets else args.target_sampler

    num_training_tasks = args.batch_size * (args.num_tasks - args.checkpoint_step - 1)
    num_testing_tasks = args.batch_size * 8

    train_loader, test_loader, test_idxs = chexpert_loader.get_chexpert_dataloader(
        args.data_path, args.batch_size, args.total_targets, args.num_targets,
        args.num_support, args.num_query, num_training_tasks, num_testing_tasks,
        uncertain_strategy=args.uncertain_cleaner, target_sampler_strategy=target_sampler_strategy,
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

    parser.add_argument('--num_support', type=int, default=6,
                        help='Total number of samples for task support dataset')
    parser.add_argument('--num_query', type=int, default=6,
                        help='Total number of samples for task support dataset')
    parser.add_argument('--num_tasks', type=int, default=4000,
                        help='Total number of tasks to sample for training (eqv: num_train_iter)')
    parser.add_argument('--total_targets', type=int, default=10,
                        help='Total number of known and unknown diseases to sample for each task')
    parser.add_argument('--num_targets', type=int, default=4,
                        help='Number of new/unknown diseases to sample for each task')
    # parser.add_argument('--num_train_iterations', type=int, default=4000,
    #                     help='number of outer-loop updates to train for')

    parser.add_argument('--test_classes', default=None, type=int, nargs='+',
                        help='Specify indices of custom test classes to hold out')
    parser.add_argument('--uncertain_cleaner', default='positive', type=str,
                        help='Specify how to replace uncertain targets in dataset')
    parser.add_argument('--target_sampler', default='known_unknown', type=str,
                        help='Specify how to sample targets for a task')

    parser.add_argument('--num_inner_steps', type=int, default=1,
                        help='number of inner-loop updates')
    parser.add_argument('--inner_lr', type=float, default=0.1,
                        help='inner-loop learning rate initialization')
    parser.add_argument('--alpha_inner_steps', type=int, default=1,
                        help='number of alpha inner-loop updates')
    parser.add_argument('--alpha_inner_lr', type=float, default=0.1,
                        help='alpha inner-loop learning rate initialization')
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