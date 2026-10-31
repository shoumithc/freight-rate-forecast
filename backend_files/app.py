# FreightRates Rate Forecast - Flask backend API
import json
import logging

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from freight_features import FEATURES, engineer_features

# ---------- configuration ----------
MODEL_PATH = "freight_rate_model.joblib"
MARKET_PATH = "market_moves.csv"
INFO_PATH = "model_info.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("freight_api")

# ---------- app + model ----------
freight_api = Flask("FreightRates Rate Forecast API")
freight_api.json.sort_keys = False   # keep batch results in row order
model = joblib.load(MODEL_PATH)      # full pipeline: preprocessing + regressor (predicts rate_move)
market = pd.read_csv(MARKET_PATH, parse_dates=["rate_date"])
market["rate_date"] = market["rate_date"].astype("datetime64[ns]")
with open(INFO_PATH) as f:
    model_info = json.load(f)
MARKET_THROUGH = market["rate_date"].max()
KNOWN_COUNTRIES = set(market["dest_country"])
logger.info("Model loaded from %s; market data through %s", MODEL_PATH, MARKET_THROUGH.date())


def forecast(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw inputs -> one row per input with the forecast price and the move vs the 4-week average."""
    features = engineer_features(raw, market)
    unknown = sorted(set(features["dest_country"]) - KNOWN_COUNTRIES)
    if unknown:
        raise ValueError(f"No market data for destination country {unknown}; known: {sorted(KNOWN_COUNTRIES)}")
    move = model.predict(features[FEATURES])
    return pd.DataFrame({
        "Predicted_Price": (move * features["price_avg_4w"]).round(2),
        "Predicted_Move_Pct": ((move - 1) * 100).round(2),
        "Price_Avg_4w": features["price_avg_4w"].round(2),
        "Market_Data_Stale": (features["rate_date"] - pd.Timedelta(weeks=1)) > MARKET_THROUGH,
    })


def error_message(e: Exception) -> str:
    """Readable message for a bad request (str() of a KeyError adds extra quotes)."""
    return str(e.args[0]) if isinstance(e, KeyError) and e.args else str(e)


@freight_api.get("/")
def home():
    """Health check / welcome route."""
    return jsonify({"message": "Welcome to the FreightRates Rate Forecast API! Use POST /v1/predict or /v1/predictbatch.",
                    **model_info})


@freight_api.post("/v1/predict")
def predict_rate():
    """Online inference: JSON payload for one lane and week -> forecast price."""
    try:
        payload = request.get_json(force=True)
        row = forecast(pd.DataFrame([payload])).iloc[0]
        logger.info("Single prediction: %.2f", row["Predicted_Price"])
        return jsonify({
            "Predicted_Price": round(float(row["Predicted_Price"]), 2),
            "Predicted_Move_Pct": round(float(row["Predicted_Move_Pct"]), 2),
            "Price_Avg_4w": round(float(row["Price_Avg_4w"]), 2),
            "Price_Unit": model_info["price_unit"],
            "Market_Data_Through": str(MARKET_THROUGH.date()),
            "Market_Data_Stale": bool(row["Market_Data_Stale"]),
        })
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": error_message(e)}), 400
    except Exception as e:  # any other failure
        logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@freight_api.post("/v1/predictbatch")
def predict_rate_batch():
    """Batch inference: CSV file (form field 'file') -> {row_index: forecast price}."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded under the key 'file'"}), 400
        batch = pd.read_csv(request.files["file"])
        result = forecast(batch)
        logger.info("Batch prediction for %d rows", len(result))
        return jsonify({str(i): round(float(p), 2) for i, p in enumerate(result["Predicted_Price"])})
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": error_message(e)}), 400
    except Exception as e:
        logger.exception("Batch prediction failed")
        return jsonify({"error": f"Batch prediction failed: {e}"}), 500


if __name__ == "__main__":
    # Local run (inside Docker, gunicorn is used instead - see Dockerfile)
    freight_api.run(host="0.0.0.0", port=7860, debug=False)
