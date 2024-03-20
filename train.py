import pandas as pd

from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load data
path = "data/properties.csv"
df = pd.read_csv(path)

# Split data into training and test sets to avoid data leakage
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# define target value
target = 'price'

selected_features = ['total_area_sqm', 'zip_code', 'latitude', 'longitude', 'nbr_bedrooms', 'surface_land_sqm']

# Change if you plan to add some categorical features too.
numerical_features = selected_features

# Numerical data preprocessing
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features)
    ])

# Create a pipeline that first preprocesses the data, then trains a model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), # First preprocesses data here
    # Training the model on the preprocessed data
    ('regressor', RandomForestRegressor(
        n_estimators=100,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features='sqrt',
        max_depth=None,
        random_state=42  # Ensuring reproducibility
    ))
])


X_train = df_train[selected_features]
y_train = df_train[target]

model_pipeline.fit(X_train, y_train)

dump(model_pipeline, 'model.joblib')
