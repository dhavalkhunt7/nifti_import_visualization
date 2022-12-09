from utilities.confusion_matrix import calc_ConfusionMatrix


def calc_Accuracy_CM(truth, pred, c=1):
    # Obtain confusion mat
    tp, tn, fp, fn = calc_ConfusionMatrix(truth, pred, c)
    # Calculate Accuracy
    acc = (tp + tn) / (tp + tn + fp + fn)
    # Return computed Accuracy
    return
