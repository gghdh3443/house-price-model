import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path

DATA_PATH = Path("data/train.csv")
MODEL_PATH = Path("models/model.joblib")

def main():
    data = pd.read_csv(DATA_PATH)

    y = data["SalePrice"]

    features = [
        "LotArea",
        "YearBuilt",
        "1stFlrSF",
        "2ndFlrSF",
        "FullBath",
        "BedroomAbvGr"
    ]

    X = data[features].copy()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, random_state=1
    )

    model = RandomForestRegressor(random_state=1)
    model.fit(X_train, y_train)

    preds = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, preds)

    print("Model MAE:", mae)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Model saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()