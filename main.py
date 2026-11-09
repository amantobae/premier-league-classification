
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score
)

# читаем датасет, encoding cp1252 нужен потому что там есть имена с умлаутами
df = pd.read_csv("PL_Players.csv", encoding="cp1252")

# переименовываем колонки, потому что работать с "Tkl" и "Int" неудобно
df = df.rename(columns={
    "Pos": "Position",
    "90s": "Total minutes played divided by 90",
    "Tkl": "Tackles",
    "TklW": "Tackles Won",
    "Press": "Number of times opponent put under pressure",
    "Succ": "Successful press",
    "Blocks": "Blocked the ball",
    "Pass": "Passes blocked",
    "Int": "Interceptions",
    "Tkl+Int": "Tackles+Interceptions",
    "Clr": "Clearances"
})

# позиции могут идти в разном порядке через запятую (напр. "FW,MF" vs "MF,FW"),
# сортируем чтобы привести к единому виду
df["Position"] = df["Position"].apply(
    lambda x: ",".join(sorted(x.split(",")))
)

# выбираем только защитные/прессинговые метрики — они лучше всего отличают позиции
features = [
    "Total minutes played divided by 90",
    "Tackles",
    "Tackles Won",
    "Number of times opponent put under pressure",
    "Successful press",
    "Blocked the ball",
    "Passes blocked",
    "Interceptions",
    "Tackles+Interceptions",
    "Clearances"
]

# убираем строки где нет данных по фичам или позиции
df = df.dropna(subset=features + ["Position"])

X = df[features]
y = df["Position"]

print("\n========== PART I ==========")

# 1. смотрим баланс классов — важно понять не перекошен ли датасет

print("\nClass distribution:")
print(y.value_counts())

print("\nClass distribution in percentages:")
print((y.value_counts(normalize=True) * 100).round(2))


# 2. делим на три выборки: 70% трейн, 15% валидация, 15% тест
# stratify=y чтобы пропорции классов сохранились в каждой выборке

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDataset sizes:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))


# 3. подбираем гиперпараметры дерева через GridSearch
# перебираем глубину, min_samples и критерий разбиения

decision_tree = DecisionTreeClassifier(random_state=42)

param_grid = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "criterion": ["gini", "entropy"]
}

grid_search = GridSearchCV(
    decision_tree,
    param_grid,
    cv=3,
    scoring="f1_macro",   # macro потому что классы примерно сбалансированы
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_tree = grid_search.best_estimator_

print("\nBest Decision Tree parameters:")
print(grid_search.best_params_)


# 4. оцениваем дерево на валидации и тесте

y_val_pred = best_tree.predict(X_val)
y_test_pred_tree = best_tree.predict(X_test)

print("\nDecision Tree validation Macro F1:")
print(round(f1_score(y_val, y_val_pred, average="macro",
                     zero_division=0), 2))

print("\nBest Model: Decision Tree")
print("Test Accuracy:",
      round(accuracy_score(y_test, y_test_pred_tree), 2))

print("Test Macro F1:",
      round(f1_score(y_test, y_test_pred_tree,
                     average="macro", zero_division=0), 2))

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_test_pred_tree,
                            zero_division=0))

# 5. KNN для сравнения — оборачиваем в pipeline со скейлером,
# потому что KNN чувствителен к масштабу признаков

knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=8))
])

knn_model.fit(X_train, y_train)

y_test_pred_knn = knn_model.predict(X_test)

knn_accuracy = accuracy_score(y_test, y_test_pred_knn)
knn_f1 = f1_score(y_test, y_test_pred_knn,
                  average="macro", zero_division=0)

tree_accuracy = accuracy_score(y_test, y_test_pred_tree)
tree_f1 = f1_score(y_test, y_test_pred_tree,
                   average="macro", zero_division=0)

print("\n========== MODEL COMPARISON ==========")

comparison = pd.DataFrame({
    "Model": ["KNN (n=8)", "Decision Tree"],
    "Accuracy": [knn_accuracy, tree_accuracy],
    "Macro F1": [knn_f1, tree_f1]
})

print(comparison.round(2))

print("\n========== PART II ==========")

# 1. оставляем только нападающих и полузащитников —
# это бинарная классификация, так что остальных убираем

binary_df = df[df["Position"].isin(["FW", "MF"])].copy()

# MF = 0, FW = 1 — произвольный выбор, просто нужна числовая метка
binary_df["Target"] = binary_df["Position"].map({
    "MF": 0,
    "FW": 1
})

X_binary = binary_df[features]
y_binary = binary_df["Target"]

print("\nFW and MF class distribution:")
print(binary_df["Position"].value_counts())

print("\nClass distribution in percentages:")
print((binary_df["Position"].value_counts(
    normalize=True
) * 100).round(2))


# 2. делим 80/20 — валидации здесь нет, она не нужна для ROC анализа

X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_binary,
    y_binary,
    test_size=0.20,
    random_state=42,
    stratify=y_binary
)


# 3. логистическая регрессия — хороший выбор для бинарки,
# даёт вероятности нужные для ROC
# class_weight="balanced" т.к. FW игроков заметно меньше

binary_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

binary_model.fit(X_train_bin, y_train_bin)

# берём вероятность принадлежности к классу FW (индекс 1)
y_prob = binary_model.predict_proba(X_test_bin)[:, 1]


# 4. строим ROC кривую и считаем AUC

fpr, tpr, thresholds = roc_curve(y_test_bin, y_prob)

auc = roc_auc_score(y_test_bin, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: FW vs MF")
plt.legend()
plt.grid()
plt.show()

print("\nROC AUC:", round(auc, 2))


# 5. смотрим как меняются метрики при разных порогах классификации
# по умолчанию порог 0.5, но иногда имеет смысл его сдвинуть

print("\nThreshold comparison:")

threshold_values = [0.3, 0.4, 0.5, 0.6, 0.7]

for threshold in threshold_values:

    y_pred_threshold = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_test_bin, y_pred_threshold)
    precision = precision_score(
        y_test_bin, y_pred_threshold, zero_division=0
    )
    recall = recall_score(
        y_test_bin, y_pred_threshold, zero_division=0
    )
    f1 = f1_score(
        y_test_bin, y_pred_threshold, zero_division=0
    )

    print("\nThreshold:", threshold)
    print("Accuracy:", round(accuracy, 2))
    print("Precision:", round(precision, 2))
    print("Recall:", round(recall, 2))
    print("F1-score:", round(f1, 2))
