import numpy as np
import pandas as pd
import torch

### CONFIGURATION ###################
NUM_FEATURES = 2
HIDDEN_SIZE = 64
NUM_LAYERS = 2
#####################################


class LSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = torch.nn.LSTM(
            NUM_FEATURES, HIDDEN_SIZE, NUM_LAYERS, batch_first=True, proj_size=2
        )


    def forward(self, x, lengths):
        # pack padding so that all sequences are the same length
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        output, _ = self.lstm(packed)

        return torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)


class NormalDataset(torch.utils.data.Dataset):
    """
    Splits up the data into normal (non-anomalous) subsequences of length at
    least 2. This is useful since the regression model needs to predict what
    normal values would be like.
    """
    def __init__(self, data):
        is_anomaly = data["is_anomaly"]
        self.x = []
        self.actual = []

        start = 0
        for end in np.append(
            np.nonzero(is_anomaly.to_numpy())[0], len(is_anomaly)
        ):
            if end - start >= 2:
                self.x.append(
                    torch.stack(
                        [
                            torch.tensor(
                                data["value"][start:end - 1].to_numpy()
                            ),
                            torch.tensor(
                                train_data["predicted"][start:end - 1]
                                .to_numpy(),
                            ),
                        ],
                        dim=1,
                    ),
                )
                self.actual.append(
                    torch.stack(
                        [
                            torch.tensor(
                                data["value"][start + 1:end].to_numpy()
                            ),
                            torch.tensor(
                                train_data["predicted"][start + 1:end]
                                .to_numpy(),
                            ),
                        ],
                        dim=1,
                    ),
                )

            start = end + 1


    def __len__(self):
        return len(self.x)


    def __getitem__(self, idx):
        return {
            "x": self.x[idx],
            "actual": self.actual[idx],
            "lengths": len(self.x[idx]),
        }


def collate_batch(batch):
    x = []
    actual = []
    lengths = []

    for item in batch:
        x.append(item["x"])
        actual.append(item["actual"])
        lengths.append(item["lengths"])

    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    actual = torch.nn.utils.rnn.pad_sequence(actual, batch_first=True)

    return {
        "x": x,
        "actual": actual,
        "lengths": torch.tensor(lengths),
    }


def trainOneEpoch(lstm, dataloader, optimizer, loss_function):
    total_loss = 0
    total_count = 0
    lstm.train()

    for batch in dataloader:
        lengths = batch["lengths"]

        predictions = lstm(batch["x"], lengths)[0]

        loss = loss_function(predictions, batch["actual"])

        # Reset the gradients in the network to zero
        optimizer.zero_grad()

        # Backprop the errors from the loss on this iteration
        loss.backward()

        # Do a weight update step
        optimizer.step()

        total_loss += loss.item()
        total_count += sum(lengths).item() * 2

    print("Average train set loss:", total_loss / total_count)


def evaluate(lstm, dataloader, loss_function):
    total_loss = 0
    total_count = 0
    lstm.eval()

    with torch.no_grad():
        for batch in dataloader:
            lengths = batch["lengths"]

            predictions = lstm(batch["x"], lengths)[0]

            loss = loss_function(predictions, batch["actual"])

            total_loss += loss.item()
            total_count += sum(lengths).item() * 2

    return total_loss / total_count


if __name__ == '__main__':
    # load datasets
    print("LOADING data from CSV")
    train_data = pd.read_csv("../dataset/processed/training.csv")
    dev_data = pd.read_csv("../dataset/processed/validation.csv")

    # instantiate model
    lstm = LSTM()
    lstm.double()

    # define optimizer and loss function
    optimizer = torch.optim.Adam(lstm.parameters())
    loss_function = torch.nn.MSELoss(reduction='sum')

    train_dataset = NormalDataset(train_data)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, collate_fn=collate_batch
    )

    # use the first half of the normal dev data as the dev set for the LSTM
    split = np.array_split(dev_data, 2)
    dev_data_1 = split[0]
    dev_dataset_1 = NormalDataset(dev_data_1)
    dev_dataloader_1 = torch.utils.data.DataLoader(
        dev_dataset_1, batch_size=1, shuffle=True, collate_fn=collate_batch
    )

    epoch = 0
    dev_set_loss = evaluate(lstm, dev_dataloader_1, loss_function)
    print ("Epoch:", epoch)
    print ("Dev set loss:", dev_set_loss)

    while True:
        trainOneEpoch(lstm, train_dataloader, optimizer, loss_function)

        epoch += 1
        dev_set_loss = evaluate(lstm, dev_dataloader_1, loss_function)
        print ("Epoch:", epoch)
        print ("Dev set loss:", dev_set_loss)
