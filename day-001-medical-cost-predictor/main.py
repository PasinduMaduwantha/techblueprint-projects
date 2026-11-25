"""
MLBlueprint — Day 001: Medical Insurance Cost Predictor
Dataset : Kaggle "Medical Cost Personal Datasets" (mirichoi0218/insurance)
Model   : RandomForestRegressor (scikit-learn)
Lesson  : Turning categorical text features into numbers & inspecting feature importance
"""

from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent / "data" / "insurance.csv"


def main():
    # 1. Load real Kaggle dataset
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # 2. Clean duplicates/missing values & encode text columns into numbers
    df = df.drop_duplicates().dropna().copy()
    df["smoker_num"] = (df["smoker"] == "yes").astype(int)
    df["sex_num"] = (df["sex"] == "male").astype(int)
    print(f"After cleaning: {len(df):,} rows")

    # 3. Quick pattern check: Smoker vs Non-Smoker average charges
    avg_non_smoker = df[df["smoker"] == "no"]["charges"].mean()
    avg_smoker = df[df["smoker"] == "yes"]["charges"].mean()
    print(f"Avg Non-Smoker Cost: ${avg_non_smoker:,.0f}/yr")
    print(f"Avg Smoker Cost    : ${avg_smoker:,.0f}/yr ({avg_smoker / avg_non_smoker:.1f}x higher!)")

    # 4. Split 80% Train / 20% Test and train a Random Forest Regressor
    features = ["age", "bmi", "children", "smoker_num", "sex_num"]
    X = df[features]
    y = df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    # 5. Evaluate on unseen test data
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"\nModel Evaluation (Test Set):")
    print(f"  R² Score : {r2:.2f} ({r2 * 100:.0f}%)")
    print(f"  MAE      : ${mae:,.0f}")

    # Feature importance ranking
    print("\nFeature Importances:")
    for name, weight in sorted(
        zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True
    ):
        print(f"  {name.replace('_num', ''):<10}: {weight * 100:5.1f}%")

    # 6. Predict on a new patient profile (Non-Smoker vs Smoker comparison)
    patient_non_smoker = pd.DataFrame(
        [{"age": 35, "bmi": 27.5, "children": 1, "smoker_num": 0, "sex_num": 1}]
    )
    patient_smoker = patient_non_smoker.copy()
    patient_smoker["smoker_num"] = 1

    pred_ns = model.predict(patient_non_smoker)[0]
    pred_sm = model.predict(patient_smoker)[0]
    print("\nLive Prediction (Age 35, BMI 27.5, 1 child, Male):")
    print(f"  Non-Smoker : ${pred_ns:,.0f}/yr")
    print(f"  Smoker     : ${pred_sm:,.0f}/yr (+${pred_sm - pred_ns:,.0f}/yr impact!)")


if __name__ == "__main__":
    main()
