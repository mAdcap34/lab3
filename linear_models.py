from __future__ import annotations

import numpy as np

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)

def train_linear_regression(data: RegressionInput) -> RegressionResult:
    features_train = np.asarray(data.train_features, dtype=float)
    target_train = np.asarray(data.train_target, dtype=float).flatten()
    features_test = np.asarray(data.test_features, dtype=float)

    rows, cols = features_train.shape

    weights = np.zeros(cols, dtype=float)
    bias = 0.0

    step_size = data.learning_rate
    num_epochs = data.epochs
    history_mse = []

    for _ in range(num_epochs):
        predictions_train = features_train @ weights + bias
        residual = predictions_train - target_train

        current_mse = float(np.mean(np.square(residual)))
        history_mse.append(current_mse)

        grad_w = (2.0 / rows) * (features_train.T @ residual)
        grad_b = (2.0 / rows) * np.sum(residual)

        weights -= step_size * grad_w
        bias -= step_size * grad_b

    test_predictions = features_test @ weights + bias

    return RegressionResult(
        weights=weights,
        bias=float(bias),
        loss_history=history_mse,
        predictions=test_predictions,
    )

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z_bounded = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_bounded))

def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    features_train = np.asarray(data.train_features, dtype=float)
    target_train = np.asarray(data.train_target, dtype=float).flatten()
    features_test = np.asarray(data.test_features, dtype=float)

    rows, cols = features_train.shape

    weights = np.zeros(cols, dtype=float)
    bias = 0.0

    step_size = data.learning_rate
    num_epochs = data.epochs
    history_bce: list[float] = []
    epsilon = 1e-12

    for _ in range(num_epochs):
        linear_out = features_train @ weights + bias
        probabilities_train = _sigmoid(linear_out)

        current_bce = float(-np.mean(
            target_train * np.log(probabilities_train + epsilon) + 
            (1.0 - target_train) * np.log(1.0 - probabilities_train + epsilon)
        ))
        history_bce.append(current_bce)

        residual = probabilities_train - target_train
        grad_w = (1.0 / rows) * (features_train.T @ residual)
        grad_b = (1.0 / rows) * np.sum(residual)

        weights -= step_size * grad_w
        bias -= step_size * grad_b

    test_proba = _sigmoid(features_test @ weights + bias)
    test_labels = np.where(test_proba >= 0.5, 1, 0)

    return ClassificationResult(
        weights=weights,
        bias=float(bias),
        loss_history=history_bce,
        probabilities=test_proba,
        predictions=test_labels,
    )

def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    )

    features_train = np.asarray(data.train_features, dtype=float)
    target_train = np.asarray(data.train_target, dtype=float).flatten()
    features_test = np.asarray(data.test_features, dtype=float)
    target_test = np.asarray(data.test_target, dtype=float).flatten()

    def evaluate(y_true, y_pred, y_proba) -> ClassificationMetrics:
        return ClassificationMetrics(
            roc_auc=float(roc_auc_score(y_true, y_proba)),
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
        )

    model_lr = LogisticRegression(max_iter=2000, random_state=42)
    model_lr.fit(features_train, target_train)
    pred_lr = model_lr.predict(features_test)
    proba_lr = model_lr.predict_proba(features_test).take(1, axis=1)

    model_dt = DecisionTreeClassifier(random_state=42)
    model_dt.fit(features_train, target_train)
    pred_dt = model_dt.predict(features_test)
    proba_dt = model_dt.predict_proba(features_test).take(1, axis=1)

    model_knn = KNeighborsClassifier(n_neighbors=5)
    model_knn.fit(features_train, target_train)
    pred_knn = model_knn.predict(features_test)
    proba_knn = model_knn.predict_proba(features_test).take(1, axis=1)

    return BaselineComparisonResult(
        logistic_regression=evaluate(target_test, pred_lr, proba_lr),
        decision_tree=evaluate(target_test, pred_dt, proba_dt),
        knn=evaluate(target_test, pred_knn, proba_knn),
    )

if __name__ == "__main__":
    import json
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    )

    raw_sp = pd.read_csv("Student_Performance.csv")
    raw_sp["Extracurricular Activities"] = raw_sp["Extracurricular Activities"].replace({"Yes": 1, "No": 0})

    matrix_x = raw_sp.drop(columns=["Performance Index"]).to_numpy(dtype=float)
    vector_y = raw_sp["Performance Index"].to_numpy(dtype=float)

    x_train, x_test, y_train, y_test = train_test_split(
        matrix_x, vector_y, test_size=0.2, random_state=42
    )

    sc_reg = StandardScaler()
    x_train_scaled = sc_reg.fit_transform(x_train)
    x_test_scaled = sc_reg.transform(x_test)

    fitted_reg = train_linear_regression(
        RegressionInput(
            train_features=x_train_scaled,
            train_target=y_train,
            test_features=x_test_scaled,
            test_target=y_test,
            learning_rate=0.05,
            epochs=800,
        )
    )

    final_mse = float(np.mean(np.square(fitted_reg.predictions - y_test)))
    print(f"[LinearRegression] Test Error (MSE) = {final_mse:.4f}")
    print(f"[LinearRegression] Train Error at Last Epoch = {fitted_reg.loss_history[-1]:.4f}")

    raw_de = pd.read_csv("german.csv", sep=";")
    vector_y_cls = raw_de["Creditability"].to_numpy(dtype=float)
    matrix_x_cls = raw_de.drop(columns=["Creditability"]).to_numpy(dtype=float)

    x_tr_c, x_te_c, y_tr_c, y_te_c = train_test_split(
        matrix_x_cls, vector_y_cls, test_size=0.25, random_state=42, stratify=vector_y_cls
    )

    sc_clf = StandardScaler()
    x_tr_c_scaled = sc_clf.fit_transform(x_tr_c)
    x_te_c_scaled = sc_clf.transform(x_te_c)

    fitted_clf = train_logistic_classifier(
        ClassificationInput(
            train_features=x_tr_c_scaled,
            train_target=y_tr_c,
            test_features=x_te_c_scaled,
            test_target=y_te_c,
            learning_rate=0.1,
            epochs=1000,
        )
    )

    def extract_scores(y_true, y_pred, y_proba):
        return {
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

    scores_own = extract_scores(y_te_c, fitted_clf.predictions, fitted_clf.probabilities)
    print(f"[Classification / Custom Logistic] {scores_own}")

    baseline_results = compare_classical_models(
        ClassificationInput(
            train_features=x_tr_c_scaled,
            train_target=y_tr_c,
            test_features=x_te_c_scaled,
            test_target=y_te_c,
        )
    )

    def unpack_metrics(obj):
        return {
            "roc_auc": float(obj.roc_auc),
            "accuracy": float(obj.accuracy),
            "precision": float(obj.precision),
            "recall": float(obj.recall),
            "f1": float(obj.f1),
        }

    scores_lr = unpack_metrics(baseline_results.logistic_regression)
    scores_dt = unpack_metrics(baseline_results.decision_tree)
    scores_knn = unpack_metrics(baseline_results.knn)

    print(f"[Classification / Sklearn Logistic] {scores_lr}")
    print(f"[Classification / Sklearn Tree]     {scores_dt}")
    print(f"[Classification / Sklearn KNN]      {scores_knn}")

    output_data = {
        "linear_regression": {"test_mse": final_mse},
        "classification": {
            "own_logistic": scores_own,
            "logistic_regression": scores_lr,
            "decision_tree": scores_dt,
            "knn": scores_knn,
        },
    }

    with open("experiment_results.json", "w", encoding="utf-8") as file_out:
        json.dump(output_data, file_out, indent=2, ensure_ascii=False)

    print("\nДанные успешно записаны в experiment_results.json")
