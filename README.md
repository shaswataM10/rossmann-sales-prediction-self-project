# Rossmann-Sales-Forecast

Predicts daily sales for a Rossmann drug store from store metadata, promotions and calendar features, using an XGBoost regressor trained on the [Kaggle Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales/data) dataset (1,115 stores, 2013–2015).

Please visit **[link to be added after deploying — see below]** for the website

## Files

| File | What it is |
|---|---|
| `rossmann_sales_forecast.ipynb` | Full pipeline: load data → feature engineering → train/validation split → train XGBoost → evaluate → save the model |
| `app.py` | Streamlit app — enter store/date/promo details, get a predicted Sales figure |
| `xgb_model.pkl`, `scaler.pkl`, `encoder.pkl` | Trained model + fitted preprocessors, produced by the notebook and loaded directly by `app.py` |

## Run it locally

```bash
git clone https://github.com/<your-username>/Rossmann-Sales-Forecast.git
cd Rossmann-Sales-Forecast
pip install -r requirements.txt
streamlit run app.py
```

The `.pkl` files in this repo were trained on a small sample so the app works
out of the box. To retrain on the real ~1M-row Kaggle dataset: download it
(instructions in the notebook, needs a free Kaggle account), then run
`rossmann_sales_forecast.ipynb` top to bottom — it overwrites the three
`.pkl` files with a model trained on the full data.

## Deploying

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and pick this repo + `app.py`.
3. Copy the `*.streamlit.app` URL it gives you into this README's link above.

## How it works

- **Feature engineering**: `CompetitionOpen` (months since the nearest competitor opened), `Promo2Open` (months the store's recurring promo has run), `IsPromoMonth` (whether the current month is in the store's active promo interval).
- **Preprocessing**: `MinMaxScaler` on numeric columns, `OneHotEncoder` on categoricals (`DayOfWeek`, `StateHoliday`, `StoreType`, `Assortment`).
- **Model**: `XGBRegressor` (`n_estimators=300, max_depth=6, learning_rate=0.1`).
- **Evaluation**: reported on an 80/20 held-out validation split, not on training data — RMSE/MAE/R² are in the notebook's output.

## Tech stack

Python · pandas · scikit-learn · XGBoost · Streamlit
