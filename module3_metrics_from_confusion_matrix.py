"""Reproduces Table 3.2 from the confusion matrix in Figure 3.2(b)."""
TN, FP, FN, TP = 850, 50, 30, 70

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

print(f"Accuracy : {accuracy:.1%}")
print(f"Precision: {precision:.1%}")
print(f"Recall   : {recall:.1%}")
print(f"F1-score : {f1:.1%}")
