# Models card 🧠

## Table of Contents 📜

- [Model details 📝](#model-details-📝)
- [Training Data 🔢](#training-data-🔢)
- [Intended Use 🎯](#intended-use-🎯)
- [Model Architecture 📐](#model-architecture-📐)
- [Training Procedure 📈](#training-procedure-📈)
- [Evaluation 📊](#evaluation-📊)
- [Limitations 🚫](#limitations-🚫)
- [Usage 👨‍💻](#usage-👨‍💻)
- [Maintainers 👷‍♂️](#maintainers-👷‍♂️)

## Project context 🌐

As part of my journey to enhance my machine learning skills, I embarked on creating a predictive model for property prices in Belgium. Choosing the BeCode dataset over previously collected data from web scraping allowed me to challenge myself with unfamiliar data. This endeavor, while initially daunting, provided a significant learning opportunity, honing my adaptability and analytical skills in the field of data science.

## Model details 📝

- **Developer**: [Danil Zhuravlov](https://www.linkedin.com/in/danil-zhuravlov/)
- **Model Date**: March 20, 2024
- **Model Version**: 1.0
- **License**: [MIT License](LICENSE)

## Training Data 🔢

- **Dataset Composition**: The dataset consists of approximately 76,000 instances, focusing on the Belgian real estate market.

- **Selected Features**: For this model, I utilized features I deemed most influential for price prediction: total area (sqm), postal code, latitude, longitude, number of bedrooms, and surface of the land.

- **Preprocessing**: Numerical features underwent median imputation for missing values, while the most frequent value strategy was applied to categorical features. All features were standardized using StandardScaler.


## Intended Use 🎯
- **Primary Use**: This model aims to provide accurate price predictions for Belgian properties based on key characteristics such as area, location, and property type.

- **Intended Users**: Real estate professionals, property investors, and individuals seeking insights into property valuations.

- **Out-of-scope**: The model's predictions are specifically tailored for the Belgian market and may not apply accurately to other regions.

## Model Architecture 📐
- **Type**: Random Forest Regressor
- **Configuration**:
  - `n_estimators`: 100
  - `min_samples_split`: 5
  - `min_samples_leaf`: 1
  - `max_features`: None
  - `max_depth`: 20

## Training Procedure 📈
- **Software/Frameworks Used**: Scikit-learn, Pandas, Python 3.8
- **Hyperparameter Tuning**: Parameters were optimized using `RandomizedSearchCV` with a 5-fold cross-validation strategy.

## Evaluation 📊
- **Metrics**:
  - Mean Squared Error (MSE): 43177832255
  - Root Mean Squared Error (RMSE): 207792
  - R² Score: 0.72
- **Cross-validation**: 5-fold cross-validation was used, yielding an average RMSE of 247438.96 and a standard deviation of 23046.

## Limitations 🚫

The model's predictive accuracy is inherently limited to the scope of the provided dataset and may not account for all factors influencing property prices, such as market trends or unseen economic conditions. Further, the model's performance is optimized for the Belgian real estate market and may not generalize well to other regions without appropriate adjustments and retraining.

## Usage 👨‍💻

Dependencies include Python 3.8, Scikit-learn, Pandas, and Joblib. To train the model, run train.py with the dataset located at data/properties.csv. For predictions, utilize predict.py, ensuring the input aligns with the model's expected feature set.

## Maintainers 👷‍♂️

For questions or issues regarding this model, please contact [Danil Zhuravlov](https://www.linkedin.com/in/danil-zhuravlov/).
