import json
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "game_pipeline.joblib"
SCHEMA_PATH = "schema.json"

def build_default_input(schema):
    x = {}
    for c in schema["feature_cols"]:
        if c in schema["num_cols"]:
            x[c] = 0.0
        else:
            x[c] = "UNK"
    if "season_id" in x:
        x["season_id"] = 21946
    if "season_type" in x:
        x["season_type"] = "Regular Season"
    if "team_abbreviation_home" in x:
        x["team_abbreviation_home"] = "HUS"
    if "team_name_home" in x:
        x["team_name_home"] = "Toronto Huskies"
    if "team_abbreviation_away" in x:
        x["team_abbreviation_away"] = "NYK"
    if "team_name_away" in x:
        x["team_name_away"] = "New York Knicks"
    return x

def main():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    clf = joblib.load(MODEL_PATH)

    x = build_default_input(schema)
    X = pd.DataFrame([x])[schema["feature_cols"]]

    for c in schema["num_cols"]:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(np.float32)

    for c in schema["cat_cols"]:
        X[c] = X[c].astype("object")
        X[c] = X[c].where(X[c].notna(), "UNK")
        X[c] = X[c].astype(str)

    pred = clf.predict(X)[0]
    proba = float(clf.predict_proba(X)[0, 1])

    print("Prediccion wl_home:", pred)
    print("Probabilidad de W:", proba)

if __name__ == "__main__":
    main()
