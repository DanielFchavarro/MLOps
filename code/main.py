from fastapi import FastAPI,  Form, HTTPException, Depends
from fastapi.responses import JSONResponse
import json
from pydantic import BaseModel, ValidationError
from datetime import date
import joblib
from enum import Enum
import numpy as np


class ModelOptions(str, Enum):
    mod_1 = "linear_regression"
    mod_2 = "random_forest"


app = FastAPI()


class InputData(BaseModel):
    study_name: str = 'PAL0708'
    sample_number: int = 1
    region: str = 'Anvers'
    island: str = 'Torgersen'
    stage: str = 'Adult, 1 Egg'
    individual_id: str = 'N1A1'
    clutch_completion:  str = 'Yes'
    date_egg: str = '11/11/07'
    culmen_length_mm: float = 39.1
    culmen_depth_mm: float = 18.7
    flipper_length_mm: float = 181.0
    body_mass_g: int = 3750
    sex: str = 'MALE'
    delta_15: float = 8.94956	
    delta_13: float = -24.69454
    comments: str = 'Not enough blood for isotopes.'

# Modelos preentrenados
mod_1 = joblib.load("../models/lr_model.pkl")
mod_2 = joblib.load("../models/rf_model.pkl")


@app.post("/make_inference/{model}")
async def make_inference(model: ModelOptions, input_data: InputData):
    try:
        # Preprocesar los datos de entrada
        input_data.sex = 1 if input_data.sex.upper() == 'MALE' else 0
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if model == ModelOptions.mod_1:
        selected_model = mod_1
        selected_features = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "sex", "body_mass_g", "delta_15","delta_13"]
    elif model == ModelOptions.mod_2:
        selected_model = mod_2
        selected_features = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm","sex", "body_mass_g", "delta_15","delta_13"]
    else:
        raise HTTPException(status_code=400, detail="Invalid model")

    selected_input_data = {feature: getattr(input_data, feature) for feature in selected_features}
    selected_input_data = {key: int(value) if isinstance(value, np.int64) else value for key, value in selected_input_data.items()}

    prediction = selected_model.predict([list(selected_input_data.values())])[0]

    try:
        return {"model": model, "prediction": prediction, "data": selected_input_data}
    except Exception as e:
                # Manejar errores de serialización personalizados si es necesario
        error_message = f"Error during serialization: {str(e)}"
        return JSONResponse(content={"error": error_message}, status_code=500)



