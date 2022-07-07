from config.setting import *


# ========================================================
# DATASET AND NEURAL NETWORK
# ========================================================


class NumericalData(Dataset):
    def __init__(self, data, transform=None):
        # data is a numpy matrix with N rows and D columns that represents a generic numerical dataset.
        # transform is a sklearn pipeline that scales and transforms the data (e.g., StandardScaler).
        # the transform pipeline must have already been fitted to the training data when this constructor is called.
        self.data = data
        self.transform = transform

    def __len__(self):
        # gets the number of rows in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # gets a data point from the dataset as torch tensor array
        point = self.data[idx]
        if self.transform:
            point = self.transform.transform(point.reshape(1, -1))
        point = torch.from_numpy(point)
        return point.float()


class Autoencoder(nn.Module):  # dim-20-10-7-10-20-dim
    def __init__(self, activation, dim):
        # dim is the number of features of the input dataset
        # activation is the activation function (e.g., nn.ReLU(), nn.Tanh())
        super(Autoencoder, self).__init__()

        self.activation = activation
        self.dim = dim

        self.encoder = nn.Sequential(
            nn.Linear(self.dim, 20),
            self.activation,
            nn.Linear(20, 10),
            self.activation,
            nn.Dropout(0.15),
            nn.Linear(10, 7),
            self.activation
        )

        self.decoder = nn.Sequential(
            nn.Linear(7, 10),
            self.activation,
            nn.Dropout(0.15),
            nn.Linear(10, 20),
            self.activation,
            nn.Linear(20, self.dim),
            self.activation
        )

    def forward(self, x):
        z = self.encoder(x)
        z = self.decoder(z)
        return z


# ========================================================
# TRAINING AND PREDICTION
# ========================================================


def train(dataloader, model, device, optimizer, loss_fn, max_epochs=20, verbose=True):
    loss_arr = []
    loss_epoch_arr = []
    for epoch in range(max_epochs):
        for i, batch in enumerate(dataloader, 0):
            model.train()
            inputs = batch
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, inputs)
            loss.backward()
            optimizer.step()
            loss_arr.append(loss.item())
        loss_epoch_arr.append(loss.item())
        if verbose:
            print(f"Epoch {epoch + 1}/{max_epochs}. Loss: {loss.item()}.")
    return model, loss_arr, loss_epoch_arr


def predict(model, data, loss_fn, device):
    squared_errors = []
    model.eval()
    for i in range(len(data)):
        inputs = data[i]
        inputs = inputs.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, inputs)
        squared_errors.append(loss.item())
    return squared_errors


def train_autoencoder(dataloader, activation, dim, device, learning_rate=0.001, weight_decay=0.0001,
                      max_epochs=20, verbose=True):
    model = Autoencoder(activation=activation, dim=dim)
    model = model.to(device)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return train(dataloader, model, device, optimizer, criterion, max_epochs=max_epochs, verbose=verbose)


def predict_with_autoencoder(model, data, device):
    criterion = nn.MSELoss(reduction='mean')
    return predict(model, data, criterion, device)


# ========================================================
# MAIN
# ========================================================


if __name__ == '__main__':





    # Initializing data loader with training data
    dataloader = DataLoader(data, batch_size=100, shuffle=True,
                            worker_init_fn=seed_worker, generator=generator_data_loader)

    # Training
    model, loss_arr, loss_epoch_arr = train_autoencoder(dataloader, nn.Tanh(), 13, device)

    # Prediction of normal data
    print(Hrule(30, "Squared errors for first 10 elements of normal data."))
    print(predict_with_autoencoder(model, new_data, device)[:10])

    # Prediction of anomalous data
    print(Hrule(30, "Squared errors for anomalous data."))
    print(predict_with_autoencoder(model, anomalies, device))


