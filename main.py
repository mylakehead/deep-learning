import sys
import json
import getopt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

from loader import VideoDataset
from model import Model


def parse_opt(argv):
    try:
        opts, args = getopt.getopt(argv, "hc:", ["config="])
    except getopt.GetoptError:
        print('main.py -c <config_file>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('main.py -c <config_file>')
            sys.exit()
        elif opt in ("-c", "--config"):
            return arg
        else:
            print('main.py -c <config_file>')
            sys.exit()


class Config:
    def __init__(self, batch, channels, frames, height, width, unit, epochs, rate, datasets):
        self.batch = batch
        self.channels = channels
        self.frames = frames
        self.height = height
        self.width = width
        self.unit = unit
        self.epochs = epochs
        self.rate = rate
        self.video_paths = []
        self.video_labels = []

        for dataset in datasets:
            for k, v in dataset.items():
                self.video_paths.append(k)
                self.video_labels.append(v)


def parse_config(config_file):
    with open(config_file, 'r') as f:
        data = json.load(f)
        config = Config(
            data['hyperparameters']['batch'],
            data['hyperparameters']['channels'],
            data['hyperparameters']['frames'],
            data['hyperparameters']['height'],
            data['hyperparameters']['width'],
            data['hyperparameters']['unit'],
            data['hyperparameters']['epochs'],
            data['hyperparameters']['rate'],
            data['datasets']
        )
        return config


def train(m, criterion, optimizer, train_loader, device):
    m.train()

    running_loss = 0.0
    all_labels = []
    all_outputs = []

    for inputs, labels in train_loader:
        # size [1] to size [1, 1]
        labels = labels.float().view(-1, 1)
        inputs, labels = inputs.to(device), labels.to(device)
        # clear the gradients of all optimized parameters.
        optimizer.zero_grad()

        outputs = m(inputs)
        loss = criterion(outputs, labels)
        # compute the gradients of the loss function with respect to each parameter in the model
        loss.backward()
        # update the parameters of a model based on the computed gradients
        optimizer.step()

        running_loss += loss.item()
        all_outputs.append(outputs.detach().numpy())
        all_labels.append(labels.numpy())

    train_loss = running_loss / len(train_loader)
    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)
    train_accuracy = accuracy_score(all_labels, all_outputs > 0.5)
    train_auc = roc_auc_score(all_labels, all_outputs)

    return train_loss, train_accuracy, train_auc


def validate(m, criterion, data_loader, device):
    m.eval()
    val_running_loss = 0.0
    val_all_labels = []
    val_all_outputs = []

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            labels = labels.float().view(-1, 1)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = m(inputs)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item()

            val_all_outputs.append(outputs.numpy())
            val_all_labels.append(labels.numpy())

            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = val_running_loss / len(data_loader)
    val_all_outputs = np.concatenate(val_all_outputs)
    val_all_labels = np.concatenate(val_all_labels)

    val_accuracy = accuracy_score(val_all_labels, val_all_outputs > 0.5)
    val_auc = roc_auc_score(val_all_labels, val_all_outputs)

    return val_loss, val_accuracy, val_auc


def main(argv):
    config_file = parse_opt(argv)
    config = parse_config(config_file)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    x_train, x_val, y_train, y_val = train_test_split(
        config.video_paths,
        config.video_labels,
        test_size=0.2,
        random_state=42
    )

    train_dataset = VideoDataset(x_train, y_train, num_frames=config.frames, frame_size=(config.height, config.width))
    val_dataset = VideoDataset(x_val, y_val, num_frames=config.frames, frame_size=(config.height, config.width))
    train_loader = DataLoader(train_dataset, batch_size=config.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()

    m = Model(config.frames, config.height, config.width, config.unit, in_channels=config.channels)
    m.to(device)
    summary(m, input_size=(config.channels, config.frames, config.height, config.width), batch_size=config.batch)

    optimizer = optim.Adam(m.parameters(), lr=config.rate)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Each epoch consists of processing all the data once,
    # performing forward and backward passes,
    # and updating model parameters based on the computed gradients.
    train_losses = []
    train_accuracies = []
    train_auc_list = []
    val_losses = []
    val_accuracies = []
    val_auc_list = []
    for epoch in range(config.epochs):
        train_loss, train_accuracy, train_auc = train(m, criterion, optimizer, train_loader, device)
        val_loss, val_accuracy, val_auc = validate(m, criterion, val_loader, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_auc_list.append(train_auc)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_auc_list.append(val_auc)
        print(
            f"Epoch [{epoch + 1}/{config.epochs}], "
            f"train_Loss: {train_loss:.4f}, train_accuracy: {train_accuracy:.4f}, train_auc: {train_auc:.4f} "
            f"val_Loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.4f}, val_auc: {val_auc:.4f}"
        )
        # scheduler.step()

    epoch_list = range(1, config.epochs + 1)

    plt.figure(figsize=(60, 60))

    plt.subplot(3, 1, 1)
    plt.plot(epoch_list, train_losses, 'bo-', label='Training Loss')
    plt.plot(epoch_list, val_losses, 'ro-', label='Validation Loss')
    # plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(epoch_list, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epoch_list, val_accuracies, 'ro-', label='Validation Accuracy')
    # plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(epoch_list, train_auc_list, 'bo-', label='Train AUC')
    plt.plot(epoch_list, val_auc_list, 'ro-', label='Validation AUC')
    # plt.title('Training and Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()

    plt.subplots_adjust(hspace=10)
    plt.tight_layout(pad=20)
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
