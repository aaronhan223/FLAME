import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize


def ptsort(tu):
  return tu[0]

def AUPRC(pts):
  true_labels = [int(x[1]) for x in pts]
  predicted_probs = [x[0] for x in pts]
  return metrics.average_precision_score(true_labels, predicted_probs)

def f1_score(truth, pred, average):
    return metrics.f1_score(truth.cpu().numpy(),pred.cpu().numpy(),average=average)

def accuracy(truth, pred):
    return metrics.accuracy_score(truth.cpu().numpy(),pred.cpu().numpy())

def metrics_multilabel(y_true, predictions, verbose=1):
    auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
    ave_auc_micro = metrics.roc_auc_score(y_true, predictions, average="micro")
    ave_auc_macro = metrics.roc_auc_score(y_true, predictions, average="macro")
    ave_auc_weighted = metrics.roc_auc_score(y_true, predictions, average="weighted")

    auprc_scores = metrics.average_precision_score(y_true, predictions, average=None)
    ave_auprc_micro = metrics.average_precision_score(y_true, predictions, average="micro")
    ave_auprc_macro = metrics.average_precision_score(y_true, predictions, average="macro")
    ave_auprc_weighted = metrics.average_precision_score(y_true, predictions, average="weighted")

    if verbose:
        print("ave_auc_micro = {}".format(ave_auc_micro))
        print("ave_auc_macro = {}".format(ave_auc_macro))
        print("ave_auc_weighted = {}".format(ave_auc_weighted))
        print("ave_auprc_micro = {}".format(ave_auprc_micro))
        print("ave_auprc_macro = {}".format(ave_auprc_macro))
        print("ave_auprc_weighted = {}".format(ave_auprc_weighted))

    return {"auc_scores": auc_scores,
            "ave_auc_micro": ave_auc_micro,
            "ave_auc_macro": ave_auc_macro,
            "ave_auc_weighted": ave_auc_weighted,
            "auprc_scores": auprc_scores,
            "ave_auprc_micro": ave_auprc_micro,
            "ave_auprc_macro": ave_auprc_macro,
            "ave_auprc_weighted": ave_auprc_weighted}

def metrics_multiclass(y_true, predictions, verbose=1):
    auc_scores = metrics.roc_auc_score(y_true, predictions, multi_class="ovr", average=None)
    ave_auc_micro = metrics.roc_auc_score(y_true, predictions, multi_class="ovr", average="micro")
    ave_auc_macro = metrics.roc_auc_score(y_true, predictions, multi_class="ovr", average="macro")
    ave_auc_weighted = metrics.roc_auc_score(y_true, predictions, multi_class="ovr", average="weighted")

    # average_precision_score expects one-hot y_true for multiclass
    predictions_arr = np.asarray(predictions)
    n_classes = predictions_arr.shape[1] if predictions_arr.ndim == 2 else None
    if n_classes is not None and n_classes >= 2:
        y_true_bin = label_binarize(np.asarray(y_true), classes=list(range(n_classes)))
        if y_true_bin.shape[1] == 1 and n_classes == 2:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        auprc_scores = metrics.average_precision_score(y_true_bin, predictions_arr, average=None)
        ave_auprc_micro = metrics.average_precision_score(y_true_bin, predictions_arr, average="micro")
        ave_auprc_macro = metrics.average_precision_score(y_true_bin, predictions_arr, average="macro")
        ave_auprc_weighted = metrics.average_precision_score(y_true_bin, predictions_arr, average="weighted")
    else:
        auprc_scores = None
        ave_auprc_micro = None
        ave_auprc_macro = None
        ave_auprc_weighted = None

    if verbose:
        print("ave_auc_micro = {}".format(ave_auc_micro))
        print("ave_auc_macro = {}".format(ave_auc_macro))
        print("ave_auc_weighted = {}".format(ave_auc_weighted))
        print("ave_auprc_micro = {}".format(ave_auprc_micro))
        print("ave_auprc_macro = {}".format(ave_auprc_macro))
        print("ave_auprc_weighted = {}".format(ave_auprc_weighted))

    return {"auc_scores": auc_scores,
            "ave_auc_micro": ave_auc_micro,
            "ave_auc_macro": ave_auc_macro,
            "ave_auc_weighted": ave_auc_weighted,
            "auprc_scores": auprc_scores,
            "ave_auprc_micro": ave_auprc_micro,
            "ave_auprc_macro": ave_auprc_macro,
            "ave_auprc_weighted": ave_auprc_weighted}