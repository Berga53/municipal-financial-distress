"""CNN model definition and shared model-evaluation utilities."""

import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import BatchNormalization, Conv1D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential


def cnn_train(X_train, y_train, epochs, batch_size, class_weight_dict, verbose=0):

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        Conv1D(64, 3, activation='relu'),
        Dropout(0.2),
        Conv1D(128, 2, activation='relu'),
        Dropout(0.2),
        Flatten(),
        BatchNormalization(),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[keras.metrics.Precision(), keras.metrics.Recall()])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, class_weight=class_weight_dict, verbose=verbose)
    return model


def get_scores(model, X_train, X_test, predictions=None, verbose=0):

    if predictions is not None:
        return predictions

    if hasattr(model, 'predict') and not hasattr(model, 'predict_proba'):
        return model.predict(X_test, verbose=verbose).flatten(), model.predict(X_train, verbose=verbose).flatten()

    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X_test)[:, 1], model.predict_proba(X_train)[:, 1]

    if hasattr(model, 'decision_function'):
        return model.decision_function(X_test), model.decision_function(X_train)

    return np.ravel(model.predict(X_test)), np.ravel(model.predict(X_train))


def find_best_threshold(preds_train, y_train, step=0.01):
    thresholds = np.arange(0, 1 + 1e-9, step)
    y = np.asarray(y_train)
    preds_bin = preds_train[None, :] > thresholds[:, None]

    tp = (preds_bin & (y == 1)).sum(axis=1)
    fp = (preds_bin & (y == 0)).sum(axis=1)
    fn = (~preds_bin & (y == 1)).sum(axis=1)

    # Matches sklearn's binary f1_score formula (2*tp / (2*tp+fp+fn)), with
    # the same zero_division=0 convention when the denominator is 0.
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(thresholds)), where=denom > 0)

    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])


def compute_metrics(y_train_for_thresh, preds_train_for_thresh, y_test, preds_test):
    t, best_f1 = find_best_threshold(preds_train_for_thresh, y_train_for_thresh)
    preds_test_bin = (preds_test > t).astype(int)
    try:
        roc = roc_auc_score(y_test, preds_test)
    except Exception:
        roc = float('nan')
    try:
        p, r, _ = precision_recall_curve(y_test, preds_test)
        pr = auc(r, p)
    except Exception:
        pr = float('nan')
    return {
        'precision_test': precision_score(y_test, preds_test_bin),
        'recall_test': recall_score(y_test, preds_test_bin),
        'f1_test': f1_score(y_test, preds_test_bin),
        'confusion_matrix_test': confusion_matrix(y_test, preds_test_bin),
        'roc_auc_test': roc,
        'pr_auc_test': pr,
        'best_threshold': t,
        'best_f1': best_f1
    }


def evaluate_model(model, X_train, y_train, X_test, y_test, predictions=None, verbose=0):
    preds_test, preds_train = get_scores(model, X_train, X_test, predictions=predictions, verbose=verbose)
    metrics = compute_metrics(y_train, preds_train, y_test, preds_test)
    try:
        metrics['roc_curve_test'] = roc_curve(y_test, preds_test)
    except Exception:
        metrics['roc_curve_test'] = (None, None, None)
    try:
        metrics['pr_curve_test'] = precision_recall_curve(y_test, preds_test)
    except Exception:
        metrics['pr_curve_test'] = (None, None, None)
    return metrics, preds_test, preds_train
