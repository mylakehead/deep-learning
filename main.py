import sys
import json
import getopt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from torchsummary import summary

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
    def __init__(self, batch, channels, frames, height, width, unit, epochs, datasets):
        self.batch = batch
        self.channels = channels
        self.frames = frames
        self.height = height
        self.width = width
        self.unit = unit
        self.epochs = epochs
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
            data['datasets']
        )
        return config


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

    criterion = nn.BCELoss()

    m = Model(config.frames, config.height, config.width, config.unit, in_channels=config.channels)
    m.to(device)
    summary(m, input_size=(config.channels, config.frames, config.height, config.width), batch_size=config.batch)

    optimizer = optim.Adam(m.parameters(), lr=0.001)

    # Each epoch consists of processing all the data once,
    # performing forward and backward passes,
    # and updating model parameters based on the computed gradients.
    for epoch in range(config.epochs):
        m.train()

        running_loss = 0.0
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

        print(f'Epoch [{epoch + 1}/{config.epochs}], Loss: {running_loss / len(train_loader)}')

    print('Finished Training')

    m.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            labels = labels.float().view(-1, 1)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = m(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_val_loss = running_val_loss / len(val_loader)
    accuracy = correct / total
    print(f'Validation Accuracy: {100 * correct / total}%, avg_val_loss: {avg_val_loss}, accuracy: {accuracy}')


if __name__ == "__main__":
    main(sys.argv[1:])
