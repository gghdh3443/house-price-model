import argparse
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/model.joblib")

def parse_args():
    p = argparse.ArgumentParser(description="Predict house price from features.")
    p.add_argument("--lotarea", type=float, required=True, help="Lot area (sq ft)")
    p.add_argument("--yearbuilt", type=int, required=True, help="Year built")
    p.add_argument("--firstflrsf", type=float, required=True, help="1st floor area (sq ft)")
    p.add_argument("--secondflrsf", type=float, required=True, help="2nd floor area (sq ft)")
    p.add_argument("--fullbath", type=int, required=True, help="Number of full baths")
    p.add_argument("--bedrooms", type=int, required=True, help="Bedrooms above grade")
    return p.parse_args()

def main():
    args = parse_args()

    model = joblib.load(MODEL_PATH)

    new_house = pd.DataFrame([{
        "LotArea": args.lotarea,
        "YearBuilt": args.yearbuilt,
        "1stFlrSF": args.firstflrsf,
        "2ndFlrSF": args.secondflrsf,
        "FullBath": args.fullbath,
        "BedroomAbvGr": args.bedrooms
    }])

    pred = model.predict(new_house)[0]
    print(f"Predicted price: {pred:.2f}")

if __name__ == "__main__":
    main()