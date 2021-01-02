from sklearn.metrics import *
import numpy as np


def accuracy(y_scores, y_pred, y_true):
    return accuracy_score(y_true, y_pred)


def recall(y_scores, y_pred, y_true):
    return recall_score(y_true, y_pred, zero_division=0)


def precision(y_scores, y_pred, y_true):
    return precision_score(y_true, y_pred, zero_division=0)


def roc_auc(y_scores, y_pred, y_true):
    y_scores = [y[1] for y in y_scores]
    return roc_auc_score(y_true, y_scores)


def pred_lift(y_scores, y_pred, y_true):
    from collections import defaultdict
    predictions = defaultdict(int)
    ground_truth = defaultdict(int)
    for pred in y_pred:
        predictions[pred] += 1
    for truth in y_true:
        ground_truth[truth] += 1
    dist = {c: round(predictions[c] / ground_truth[c], 3) for c in sorted(ground_truth)}
    return dict(dist)


def best_threshold(y_scores, y_pred, y_true):
    y_scores = [y[1] for y in y_scores]
    positives, negatives = np.sum(y_true == 1), np.sum(y_true == 0)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    true_positives = recalls * positives
    false_positives = true_positives * (1 / np.where(precisions == 0, 1.0, precisions) - 1)
    true_negatives = negatives - false_positives
    accuracies = (true_positives + true_negatives) / (positives + negatives)
    adj_accuracies = 0.5 * true_positives / positives + 0.5 * true_negatives / negatives
    acc_max_id = np.argmax(accuracies)
    adj_max_id = np.argmax(adj_accuracies)
    best_acc_threshold = thresholds[acc_max_id]
    best_adj_threshold = thresholds[adj_max_id]
    f1s = (2 * precisions * recalls) / np.where(precisions + recalls == 0, 1.0, precisions + recalls)
    f1_max_id = np.argmax(f1s)
    best_f1_threshold = thresholds[f1_max_id]
    return {
        'best_accuracy': {'accuracy': accuracies[acc_max_id], 'adj_accuracy': adj_accuracies[acc_max_id],
                          'recall': recalls[acc_max_id], 'precision': precisions[acc_max_id], 'f1': f1s[acc_max_id],
                          'threshold': best_acc_threshold},
        'best_adj_accuracy': {'accuracy': accuracies[adj_max_id], 'adj_accuracy': adj_accuracies[adj_max_id],
                              'recall': recalls[adj_max_id], 'precision': precisions[adj_max_id], 'f1': f1s[adj_max_id],
                              'threshold': best_adj_threshold},
        'best_f1': {'accuracy': accuracies[f1_max_id], 'adj_accuracy': adj_accuracies[f1_max_id],
                    'recall': recalls[f1_max_id], 'precision': precisions[f1_max_id], 'f1': f1s[f1_max_id],
                    'threshold': best_f1_threshold}
    }