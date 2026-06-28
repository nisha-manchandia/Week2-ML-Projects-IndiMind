import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))



np.random.seed(42)

from sklearn.datasets import make_classification


X, y = make_classification(
    n_samples=768,
    n_features=8,
    n_informative=6,
    n_redundant=2,
    random_state=42
)

feature_names = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age"
]

df = pd.DataFrame(X, columns=feature_names)
df["Outcome"] = y

print("=" * 55)
print("       TASK 3 — CLASSIFICATION SPRINT")
print("=" * 55)
print("\n Dataset: Diabetes Prediction (Pima Indians style)")
print(f"   Samples : {df.shape[0]}")
print(f"   Features: {df.shape[1] - 1}")
print(f"   Target  : Outcome (0 = No Diabetes, 1 = Diabetes)")

print("\n First 5 rows:")
print(df.head().to_string(index=False))

print("\n Class distribution:")
print(df["Outcome"].value_counts().to_string())


X_data = df.drop("Outcome", axis=1).values
y_data = df["Outcome"].values

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_sc, y_train)

y_pred = model.predict(X_test_sc)
acc = accuracy_score(y_test, y_pred)

print("\n" + "=" * 55)
print("              MODEL RESULTS")
print("=" * 55)
print(f"\n Accuracy : {acc * 100:.2f}%")
print("\n Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=["No Diabetes", "Diabetes"]))


cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["No Diabetes", "Diabetes"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix — Logistic Regression\nDiabetes Prediction",
             fontsize=11, fontweight='bold', pad=12)
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n Confusion matrix saved to: {cm_path}")

fig2, ax2 = plt.subplots(figsize=(5, 3.5))
report_dict = classification_report(y_test, y_pred, output_dict=True)
metrics_vals = [acc,
                float(report_dict['1']['precision']),
                float(report_dict['1']['recall']),
                float(report_dict['1']['f1-score'])]
metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
bars = ax2.bar(metric_labels, [v * 100 for v in metrics_vals], color=colors, width=0.5)
for bar, val in zip(bars, metrics_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val*100:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.set_ylim(0, 115)
ax2.set_ylabel("Score (%)")
ax2.set_title("Model Performance Metrics\nLogistic Regression — Diabetes Prediction",
              fontsize=10, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
plt.tight_layout()
acc_path = os.path.join(OUTPUT_DIR, "accuracy_chart.png")
plt.savefig(acc_path, dpi=150, bbox_inches='tight')
plt.close()
print(f" Accuracy chart saved to: {acc_path}")
print("\n All outputs generated successfully.")