import numpy as np
import pandas as pd
import torch

from scipy.stats import norm

### CONFIGURATION ###################
BATCH_SIZE = 1
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
                                data["predicted"][start:end - 1]
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
                                data["predicted"][start + 1:end]
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


def calculateF1Score(log_likelihoods, value_threshold, is_anomaly):
    """
    Precondition: log_likelihoods and is_anomaly should be sorted by predicted
    """
    pred_is_anomaly = log_likelihoods[:,0] < value_threshold

    tp = sum(pred_is_anomaly & is_anomaly)
    fp = sum(pred_is_anomaly & np.logical_not(is_anomaly))
    fn = sum(np.logical_not(pred_is_anomaly) & is_anomaly)

    best_f1 = tp / (tp + 1 / 2 * (fp + fn))
    best_predicted_threshold = log_likelihoods[0,1]

    for i in range(1, len(is_anomaly)):
        if pred_is_anomaly[i - 1]:
            if is_anomaly[i - 1]:
                tp -= 1
                fn += 1
            else:
                fp -= 1

            f1 = tp / (tp + 1 / 2 * (fp + fn))

            if f1 > best_f1:
                best_f1 = f1
                best_predicted_threshold = log_likelihoods[i,1]

    return best_predicted_threshold, best_f1


def calculateF1ScoreFromPredictedThreshold(
    log_likelihoods, value_threshold, predicted_threshold, is_anomaly
):
    pred_is_anomaly = (
        (log_likelihoods[:,0] < value_threshold)
        & (log_likelihoods[:,1] >= predicted_threshold)
    )

    tp = sum(pred_is_anomaly & is_anomaly)
    fp = sum(pred_is_anomaly & np.logical_not(is_anomaly))
    fn = sum(np.logical_not(pred_is_anomaly) & is_anomaly)

    return tp / (tp + 1 / 2 * (fp + fn))


def calculateLogLikelihoods(lstm, x, actual, means, stds):
    lstm.eval()

    with torch.no_grad():
        predictions = lstm(x, [len(x[0])])[0]

        errors = actual - predictions

    # assume first data point is non-anomalous
    return np.concatenate(
        ([[0, float('-inf')]],
         norm.logpdf(errors, loc=means, scale=stds).squeeze()),
    )


def determineClassificationThresholds(lstm, x, actual, means, stds, is_anomaly):
    """
    Returns the best f1, value threshold, and predicted threshold
    """
    best_f1 = -1

    log_likelihoods = calculateLogLikelihoods(lstm, x, actual, means, stds)

    index_array = np.argsort(log_likelihoods, axis=0)
    indices = index_array[:,1]

    # sort log likelihoods and anomalies by predicted
    log_likelihoods_sorted = log_likelihoods[indices]
    is_anomaly_sorted = is_anomaly[indices].to_numpy()

    for value_threshold in log_likelihoods_sorted[:,0]:
        predicted_threshold, f1 = calculateF1Score(
            log_likelihoods_sorted, value_threshold, is_anomaly_sorted
        )

        if f1 > best_f1:
            best_f1 = f1
            best_value_threshold = value_threshold
            best_predicted_threshold = predicted_threshold

    print("Best f1:", best_f1)
    print("Best value threshold:", best_value_threshold)
    print("Best predicted threshold:", best_predicted_threshold)

    f1 = calculateF1ScoreFromPredictedThreshold(
        log_likelihoods,
        best_value_threshold,
        best_predicted_threshold,
        is_anomaly,
    )
    print("Sanity check with method using predicted threshold:", f1)

    return best_f1, best_value_threshold, best_predicted_threshold


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
    """
    Returns error means, standard deviations, and loss
    """
    total_loss = 0
    total_count = 0
    lstm.eval()

    errors = torch.tensor([])

    with torch.no_grad():
        for batch in dataloader:
            lengths = batch["lengths"]

            predictions = lstm(batch["x"], lengths)[0]

            loss = loss_function(predictions, batch["actual"])

            batch_errors = batch["actual"] - predictions

            for i in range(len(lengths)):
                errors = torch.cat([errors, batch_errors[i][:lengths[i]]])

            total_loss += loss.item()
            total_count += sum(lengths).item() * 2

    avg_loss = total_loss / total_count
    print("Loss:", avg_loss)

    errors_np = errors.numpy()
    means = np.mean(errors_np, axis=0)
    stds = np.std(errors_np, axis=0)
    print("Error means:", means)
    print("Error standard deviations:", stds)

    return means, stds, avg_loss


if __name__ == '__main__':
    model_file_path = input("Model file path? ")
    batch_size = int(input("Batch size? "))

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
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    # use the second half of the normal dev data to calculate error means and
    # standard deviations
    split = np.array_split(dev_data, 2)
    dev_data_1 = split[1]
    dev_dataset_1 = NormalDataset(dev_data_1)
    dev_dataloader_1 = torch.utils.data.DataLoader(
        dev_dataset_1,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    # use the first half of the dev data to determine classification thresholds
    dev_data_2 = split[0]
    is_anomaly = dev_data_2["is_anomaly"]
    dev_data_2_length = len(dev_data_2)
    dev_data_2_x = (
        torch.stack(
            [
                torch.tensor(
                    dev_data_2["value"][:dev_data_2_length - 1].to_numpy()
                ),
                torch.tensor(
                    dev_data_2["predicted"][:dev_data_2_length - 1].to_numpy()
                ),
            ],
            dim=1,
        )
        .unsqueeze(dim=0)
    )
    dev_data_2_actual = (
        torch.stack(
            [
                torch.tensor(dev_data_2["value"][1:].to_numpy()),
                torch.tensor(dev_data_2["predicted"][1:].to_numpy()),
            ],
            dim=1,
        )
        .unsqueeze(dim=0)
    )

    epoch = 0
    best_loss = float('inf')
    best_f1 = -1
    patience = 0
    print ("Epoch:", epoch)
    means, stds, loss = evaluate(lstm, dev_dataloader_1, loss_function)

    (
        f1, value_threshold, predicted_threshold
    ) = determineClassificationThresholds(
        lstm, dev_data_2_x, dev_data_2_actual, means, stds, is_anomaly
    )

    while patience < 5:
        if loss < best_loss:
            best_loss = loss
            patience = 0
        if f1 > best_f1:
            best_f1 = f1
            best_value_threshold = value_threshold
            best_predicted_threshold = predicted_threshold
            patience = 0
            best_means = means
            best_stds = stds
            # save the best model
            torch.save(lstm.state_dict(), model_file_path)

        trainOneEpoch(lstm, train_dataloader, optimizer, loss_function)

        epoch += 1
        patience += 1
        print("Epoch:", epoch)
        means, stds, loss = evaluate(lstm, dev_dataloader_1, loss_function)
        (
            f1, value_threshold, predicted_threshold
        ) = determineClassificationThresholds(
            lstm, dev_data_2_x, dev_data_2_actual, means, stds, is_anomaly
        )

    print("Final best f1:", best_f1)
    print("Final best value threshold:", best_value_threshold)
    print("Final best predicted threshold:", best_predicted_threshold)
    print("Final best means:", best_means)
    print("Final best stds:", best_stds)

    print("EVALUATION")

    # reload the best model
    lstm = LSTM()
    lstm.double()
    lstm.load_state_dict(torch.load(model_file_path))

    print("Final dev set 2 evaluation")
    log_likelihoods = calculateLogLikelihoods(
        lstm, dev_data_2_x, dev_data_2_actual, best_means, best_stds
    )
    f1 = calculateF1ScoreFromPredictedThreshold(
        log_likelihoods,
        best_value_threshold,
        best_predicted_threshold,
        is_anomaly.to_numpy(),
    )
    print("F1:", f1)
