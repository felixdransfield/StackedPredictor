from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, \
    brier_score_loss, auc, confusion_matrix


def perf_measure(y_true, y_pred):

    CM = confusion_matrix(y_true, y_pred)

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]

    return(TP, FP, TN, FN)

def performance_metrics(testing_y, y_pred_binary):
    F1Macro = f1_score(testing_y, y_pred_binary, average='macro')
    PrecisionMacro = precision_score(testing_y, y_pred_binary, average='macro')
    RecallMacro = recall_score(testing_y, y_pred_binary, average='macro')
    Accuracy = accuracy_score(testing_y, y_pred_binary)
    ClassificationReport = classification_report(testing_y, y_pred_binary)
    cf = confusion_matrix(testing_y, y_pred_binary)
    #PRAUC = auc(RecallMacro, PrecisionMacro)

    #BrierScoreProba = brier_score_loss(testing_y, y_pred_rt)
    #BrierScoreBinary = brier_score_loss(testing_y, y_pred_binary)

    performance_row = {
        "F1-Macro" : F1Macro,
        "Precision-Macro" : PrecisionMacro,
        "Recall-Macro" : RecallMacro,
        "Accuracy" : Accuracy,
        "ClassificationReport" : ClassificationReport,
        "ConfusionMatrix": cf
        #"Precision-Recall AUC": PRAUC
        #"BrierScoreProba" : BrierScoreProba,
        #"BrierScoreBinary" : BrierScoreBinary
    }

    return performance_row
