from pathlib import Path
import pickle

import typer
import pandas as pd
import torch
import torch.nn as nn

app = typer.Typer(
    short_help='Materials prediction',
    pretty_exceptions_enable=False,
    add_completion=False,
    rich_help_panel=None,
)


def get_model():
    net = nn.Sequential(
        nn.Linear(12, 16),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(16, 1)
    )
    return net


@app.command()
def main(
        path_to_weights: Path = Path('net.pth'),
        path_to_data: Path = Path('test_x.csv'),
        path_to_scaler_x: Path = Path('x_scaler.pkl'),
        path_to_scaler_y: Path = Path('y_scaler.pkl'),
):
    """
    Predict composite materials.

    :param path_to_weights: Path
        Path to trained neural network weights.
    :param path_to_data: Path
         Path to CSV file with test data.
    :param path_to_scaler_x: Path
        Saved StandardScaler for features.
    :param path_to_scaler_y: Path
        Saved StandardScaler for predictions.
    """
    df = pd.read_csv(path_to_data, index_col=None)
    with open(path_to_scaler_x, 'rb') as f:
        scaler_x = pickle.load(f)
    with open(path_to_scaler_y, 'rb') as f:
        scaler_y = pickle.load(f)

    net = get_model()
    net.load_state_dict(torch.load(path_to_weights))
    net = net.cpu()
    net.eval()

    test_x_scaled = scaler_x.transform(df)
    with torch.no_grad():
        predicted = net(torch.Tensor(test_x_scaled)).numpy()
    predicted = scaler_y.inverse_transform(predicted).ravel()

    print('Предсказанные значения:')
    for p in predicted:
        print(p)


if __name__ == '__main__':
    app()