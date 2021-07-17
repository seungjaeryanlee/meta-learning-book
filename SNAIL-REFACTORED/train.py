# coding=utf-8
"""
Main script for training SNAIL on Omniglot.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from networks import SnailFewShot

from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader


def get_acc(last_model, last_targets):
    _, preds = last_model.max(1)
    acc = torch.eq(preds, last_targets).float().mean()
    return acc.item()


def process_torchmeta_batch(batch, options):
    """
    Process batch from torchmeta dataset for the SNAIL model

    Parameters
    ----------
    batch : dict
        A dictionary given by the torchmeta dataset
    options : SimpleNamespace
        A namespace with configuration details.

    Returns
    -------
    input_images : torch.tensor
        Input images to the SNAIL model of shape (batch_size*(N*K+1), img_channels, img_height,
        img_width) where N and K denote the same variables as in the N-way K-shot problem.
    input_onehot_labels : torch.tensor
        Input one-hot labels to the SNAIL model of shape (batch_size*(N*K+1), N) where N and K
        denote the same variables as in the N-way K-shot problem. The last label for each (N*K+1)
        is not used since it is the target label.
    target_labels : torch.tensor
        Test set labels to evaluate the SNAIL model by comparing with its outputs. Has shape
        (batch_size).

    """
    train_inputs, train_labels = batch["train"]
    test_inputs, test_labels = batch["test"]

    # Select one image from N images in the test set
    chosen_indices = torch.randint(test_inputs.shape[1], size=(test_inputs.shape[0],))
    chosen_test_inputs = test_inputs[torch.arange(test_inputs.shape[0]), chosen_indices, :, :, :].unsqueeze(1)
    chosen_test_labels = test_labels[torch.arange(test_labels.shape[0]), chosen_indices].unsqueeze(1)

    # Concatenate train and test set for SNAIL-style input images and labels
    input_images = torch.cat((train_inputs, chosen_test_inputs), dim=1).reshape((-1, *train_inputs.shape[2:]))
    input_labels = torch.cat((train_labels, chosen_test_labels), dim=1).reshape((-1, *train_labels.shape[2:]))

    # Convert labels to one-hot
    input_onehot_labels = F.one_hot(input_labels).float()

    # Separate out target labels
    target_labels = input_labels[::(options.num_cls * options.num_samples + 1)].long()

    # Move to correct device
    if options.cuda:
        input_images, input_onehot_labels = input_images.cuda(), input_onehot_labels.cuda()
        target_labels = target_labels.cuda()

    return input_images, input_onehot_labels, target_labels


def train(model, optimizer, train_dataloader, val_dataloader, opt):
    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    best_model_path = os.path.join(opt.exp, 'best_model.pth')
    last_model_path = os.path.join(opt.exp, 'last_model.pth')

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))

        # Training phase
        model.train()
        for i, batch in tqdm(enumerate(train_dataloader), total=1000):
            if i >= 1000:
                break
            input_images, input_onehot_labels, target_labels = process_torchmeta_batch(batch, opt)
            predicted_labels = model(input_images, input_onehot_labels)[:, -1, :]
            loss = loss_fn(predicted_labels, target_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_acc.append(get_acc(predicted_labels, target_labels))

        avg_loss = np.mean(train_loss[-opt.iterations:])
        avg_acc = np.mean(train_acc[-opt.iterations:])
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))

        # Validation phase
        model.eval()
        for i, batch in tqdm(enumerate(val_dataloader), total=1000):
            if i >= 1000:
                break
            input_images, input_onehot_labels, target_labels = process_torchmeta_batch(batch, opt)
            predicted_labels = model(input_images, input_onehot_labels)[:, -1, :]
            loss = loss_fn(predicted_labels, target_labels)

            val_loss.append(loss.item())
            val_acc.append(get_acc(predicted_labels, target_labels))

        avg_loss = np.mean(val_loss[-opt.iterations:])
        avg_acc = np.mean(val_acc[-opt.iterations:])

        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()

        # TODO(seungjaeryanlee): Understand this code better
        for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
            with open(os.path.join(opt.exp, name + '.txt'), 'w') as f:
                for item in locals()[name]:
                    f.write("%s\n" % item)

    torch.save(model.state_dict(), last_model_path)

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def test(model, test_dataloader, opt):
    """
    Test model on given dataset and options.
    """
    model.eval()
    acc_per_epoch = []
    for epoch in range(opt.test_epochs):
        for i, batch in tqdm(enumerate(test_dataloader), total=1000):
            if i >= 1000:
                break
            input_images, input_onehot_labels, target_labels = process_torchmeta_batch(batch, opt)
            predicted_labels = model(input_images, input_onehot_labels)[:, -1, :]

            acc_per_epoch.append(get_acc(predicted_labels, target_labels))

    avg_acc = np.mean(acc_per_epoch)
    print('Test Acc: {}'.format(avg_acc))

    return avg_acc

def main():
    """
    Initialize everything and train
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='default')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--test-epochs', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--dataset', type=str, default='omniglot')
    parser.add_argument('--num_cls', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cuda', action='store_true')
    options = parser.parse_args()

    if not os.path.exists(options.exp):
        os.makedirs(options.exp)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Setup dataset
    train_dataset = omniglot("data", ways=5, shots=1, test_shots=1, meta_train=True, download=True)
    train_dataloader = BatchMetaDataLoader(train_dataset, batch_size=options.batch_size, num_workers=8)
    val_dataset = omniglot("data", ways=5, shots=1, test_shots=1, meta_val=True, download=True)
    val_dataloader = BatchMetaDataLoader(val_dataset, batch_size=options.batch_size, num_workers=8)
    test_dataset = omniglot("data", ways=5, shots=1, test_shots=1, meta_test=True, download=True)
    test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=options.batch_size, num_workers=8)
    # Setup model
    model = SnailFewShot(options.num_cls, options.num_samples, options.dataset, options.cuda)
    model = model.cuda() if options.cuda else model
    # Setup optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=options.lr)

    # Train model
    train_result = train(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        opt=options,
    )
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = train_result

    # Test last model
    print('Testing with last model..')
    test(
        model=model,
        test_dataloader=test_dataloader,
        opt=options,
    )

    # Test best model
    model.load_state_dict(best_state)
    print('Testing with best model..')
    test(
        opt=options,
        test_dataloader=test_dataloader,
        model=model,
    )


if __name__ == '__main__':
    main()
