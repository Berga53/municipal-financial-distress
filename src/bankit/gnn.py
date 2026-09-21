"""Graph neural network model and its training/evaluation loop."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from .models import evaluate_model
from .utils import print_one_line


class GNN(nn.Module):
    def __init__(self, in_feats):

        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_feats, 64, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, kernel_size=2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )

        self.conv1 = GCNConv(32, 64)
        self.conv2 = GCNConv(64, 64)

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_weight=None):

        x = x.permute(0, 2, 1)

        h = self.cnn(x)

        h = F.relu(self.conv1(h, edge_index, edge_weight=edge_weight))
        h = F.relu(self.conv2(h, edge_index, edge_weight=edge_weight))

        out = self.fc(h)
        return out


def gnn_train_evaluate(X, y, adj, edge_weight, n_munis, num_iter=200):

    print_one_line('Preparing train/test split...')

    X = X[:n_munis]
    y = y[:n_munis]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_nodes, t, f = X.shape

    train_idx, test_idx = train_test_split(np.arange(num_nodes), test_size=0.2, random_state=42, stratify=y)

    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)

    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)

    edge_index = adj.nonzero(as_tuple=False).t().contiguous()
    edge_index = edge_index.to(device)
    edge_attr = edge_weight.to(device)

    data = Data(x=X.to(device), edge_index=edge_index, edge_attr=edge_attr, y=y.to(device))
    data = data.to(device)

    y_train_np = y[train_idx].cpu().numpy()
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_np), y=y_train_np)

    class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
    sample_weights = class_weights[y[train_idx]]

    model = GNN(in_feats=f).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCELoss(weight=sample_weights)

    print_one_line('Training GNN...')

    for _ in range(num_iter):
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.edge_attr).squeeze(1)

        loss = criterion(out[train_idx], data.y[train_idx].float())

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr).cpu().numpy()

    predictions_test = out[test_idx]
    predictions_train = out[train_idx]
    predictions = (predictions_test, predictions_train)

    print_one_line('Evaluating GNN...')

    metrics, _, _ = evaluate_model(model, X[train_idx], y[train_idx], X[test_idx], y[test_idx], predictions, verbose=0)

    return model, metrics
