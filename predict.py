import pandas as pd
from joblib import load

# Load the trained model
model = load('model.joblib')


new_data = {
    'total_area_sqm': [90],
    'surface_land_sqm': [0],
    'latitude': [50.97990981194803],
    'longitude': [3.5300126547156063],
    'nbr_bedrooms': [2],
    'zip_code': [9800],
    
}

df_new = pd.DataFrame(new_data)

# Predict using the loaded model
predicted_price = model.predict(df_new)

print(f"Predicted Price: {round(predicted_price[0])}€")
