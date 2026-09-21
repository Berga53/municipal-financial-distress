"""Top-level training loop that fits and evaluates every model family."""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from .gnn import gnn_train_evaluate
from .graph import prepare_graph
from .models import cnn_train, evaluate_model
from .utils import print_one_line


def run_training(X, y, comuni, epochs, batch_size, reshape=True, verbose=0, random_state=42):

    print_one_line('Preparing train/test split...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    cw = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {0: cw[0], 1: cw[1]}
    class_sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)

    if reshape:
        X_train_reshaped = np.concatenate([X_train[:, :, :-5].reshape(X_train.shape[0], -1), X_train[:, 0, -5:]], axis=-1)
        X_test_reshaped = np.concatenate([X_test[:, :, :-5].reshape(X_test.shape[0], -1), X_test[:, 0, -5:]], axis=-1)
    else:
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

    print_one_line('Training CNN...')
    cnn_model = cnn_train(X_train, y_train, epochs, batch_size, class_weight_dict, verbose)

    print_one_line('Evaluating CNN...')
    cnn_metrics, _, _ = evaluate_model(cnn_model, X_train, y_train, X_test, y_test, verbose=verbose)

    print_one_line('Preparing graph for GNN...')
    adj, edge_weight = prepare_graph(comuni)

    gnn_model, gnn_metrics = gnn_train_evaluate(X, y, adj, edge_weight, n_munis=comuni.shape[0], num_iter=epochs)

    models = {
        "Logistic Regression": LogisticRegression(penalty='l2', C=1, solver='sag', class_weight='balanced', max_iter=10000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=None, random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced'),
        "XGBoost": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)}

    model_objects = {"CNN": cnn_model, "GNN": gnn_model}
    results = {"CNN": cnn_metrics, "GNN": gnn_metrics}

    for model_name, model in models.items():
        print_one_line(f"Training {model_name}...")

        sample_weight = class_sample_weight if model_name == "XGBoost" else None
        try:
            model.fit(X_train_reshaped, y_train, sample_weight=sample_weight)
        except TypeError:
            model.fit(X_train_reshaped, y_train)

        print_one_line(f"Evaluating {model_name}...")
        metrics, _, _ = evaluate_model(model, X_train_reshaped, y_train, X_test_reshaped, y_test, verbose=verbose)

        model_objects[model_name] = model
        results[model_name] = metrics
    print_one_line('Done.')

    return model_objects, results
