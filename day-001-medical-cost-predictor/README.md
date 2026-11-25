# Day 001 — Medical Insurance Cost Predictor

Predict annual medical insurance charges from patient attributes using a **Random Forest Regressor** on Kaggle's classic **Medical Cost Personal Dataset**.

## Practical ML Takeaways
1. **Clean before modeling**: Always check for duplicate records (`1,338 → 1,337`) and missing values.
2. **Encode categorical features**: ML models read numbers — turning `"yes"/"no"` into `1/0` unlocks the strongest predictor in the dataset.
3. **Inspect feature importance**: Tree models tell you *why* they make predictions (`smoker` alone explains **63.0%** of cost variance, followed by `BMI` at **21.1%** and `age` at **13.4%**).

## Dataset
- **Source**: Kaggle — [Medical Cost Personal Datasets (`mirichoi0218/insurance`)](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Records**: `1,338` raw rows → `1,337` clean rows
- **Features**: `age`, `bmi`, `children`, `smoker`, `sex`
- **Target**: `charges` (annual USD medical cost)

## Results (80/20 Train-Test Split)

| Metric | Value |
| :--- | :--- |
| **Test $R^2$ Score** | **0.89 (89%)** |
| **Mean Absolute Error (MAE)** | **$2,536** |
| **Sample Prediction (35yo, BMI 27.5, 1 child, Male, Non-Smoker)** | **$5,611 / yr** |
| **Same Profile (Smoker)** | **$19,957 / yr** (+$14,346 / yr) |

## Run Locally

```bash
pip install -r requirements.txt
python main.py
```
