from sklearn import metrics


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
    # import pdb; pdb.set_trace()
    auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
    ave_auc_micro = metrics.roc_auc_score(y_true, predictions,
                                          average="micro")
    ave_auc_macro = metrics.roc_auc_score(y_true, predictions,
                                          average="macro")
    ave_auc_weighted = metrics.roc_auc_score(y_true, predictions,
                                             average="weighted")

    if verbose:
        # print("ROC AUC scores for labels:", auc_scores)
        print("ave_auc_micro = {}".format(ave_auc_micro))
        print("ave_auc_macro = {}".format(ave_auc_macro))
        print("ave_auc_weighted = {}".format(ave_auc_weighted))

    return{"auc_scores": auc_scores,
            "ave_auc_micro": ave_auc_micro,
            "ave_auc_macro": ave_auc_macro,
            "ave_auc_weighted": ave_auc_weighted}