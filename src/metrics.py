from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def print_report(y_true, y_pred, name: str = "") -> None:
    m = compute_metrics(y_true, y_pred)
    print(f"--- {name} ---")
    print(f"precision: {m['precision']:.3f}")
    print(f"recall:    {m['recall']:.3f}  <- attacks caught out of all real attacks")
    print(f"f1:        {m['f1']:.3f}")
    print("confusion matrix [[TN FP] [FN TP]]:")
    print(m["confusion_matrix"])