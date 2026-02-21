import gradio as gr
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/model.joblib")
model = joblib.load(MODEL_PATH)

def predict(lotarea, yearbuilt, firstflrsf, secondflrsf, fullbath, bedrooms):
    X = pd.DataFrame([{
        "LotArea": lotarea,
        "YearBuilt": yearbuilt,
        "1stFlrSF": firstflrsf,
        "2ndFlrSF": secondflrsf,
        "FullBath": fullbath,
        "BedroomAbvGr": bedrooms,
    }])
    price = model.predict(X)[0]
    return float(price)

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="LotArea (sq ft)", value=8000),
        gr.Number(label="YearBuilt", value=2005),
        gr.Number(label="1stFlrSF (sq ft)", value=1200),
        gr.Number(label="2ndFlrSF (sq ft)", value=500),
        gr.Number(label="FullBath", value=2),
        gr.Number(label="Bedrooms", value=3),
    ],
    outputs=gr.Number(label="Predicted price"),
    title="House Price Predictor",
    description="Dette er en type KI-modell som er trent opp med bruk av Kraggle sitt hus-pris-datasett, og den kan brukes til å forutsi pris på et hus basert på ulike kriterier.",
)

if __name__ == "__main__":
    demo.launch()