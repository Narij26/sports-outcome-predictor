import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

from preprocess import preprocess

DATA_PATH = os.path.join("data", "nba_games.csv")
MODEL_PATH = os.path.join("models", "nba_rf_model.joblib")


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Run: python src/fetch_data.py")

    df = pd.read_csv(DATA_PATH)
    data = preprocess(df)

    X = data.drop(columns=["home_win"])
    y = data["home_win"]

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )

    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    print(f"Holdout Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "feature_columns": list(X.columns)}, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
