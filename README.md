# Immo-Eliza Property Price Prediction 🏡

## Overview 📝

This project aims to predict property prices in Belgium using machine learning techniques. Developed as part of a skill enhancement initiative, it leverages the BeCode dataset to train a Random Forest Regressor model, focusing on features like total area, surface area, postal code, latitude, longitude, and the number of bedrooms.

## Table of Contents 🔍
- [Getting Started 🚀](#getting-started-🚀)
- [Prerequisites 🔑](#prerequisites-🔑)
- [Installation 🔧](#installation-🔧)
- [Usage 🔄](#usage-🔄)
- [Project Structure 📁](#project-structure-📁)
- [Maintainers 👷‍♂️](#maintainers-👷‍♂️)
- [License ⚖️](#license-⚖️)
- [Models Card](MODELSCARD.md)

## Getting Started 🚀

### Prerequisites 🔑
- Python 3.8+
- Pip for package installation

### Installation 🔧

    git@github.com:Danil-Zhuravlov/immo-eliza-ml.git
    cd immo-eliza-ml

**Set up a virtual environment** (optional but recommended)

    python3 -m venv venv
    source venv/bin/activate
    
On Windows use

    venv\Scripts\activate

**Install the dependencies**

    pip install -r requirements.txt

### Usage 🔄

- **Training the model**

Run the `train.py` script to train the model on the provided dataset. This script preprocesses the data, trains the Random Forest Regressor model, and saves it for future predictions.

    python3 train.py

- **Making predictions**

Use the `predict.py` script to make price predictions on new property data. The script loads the trained model and uses it to predict the price based on the input features.

    python3 predict.py

## Project Structure 📁
- `data/`: Folder containing the dataset used for training.
- `train.py`: Script for training the machine learning model.
- `predict.py`: Script for making predictions using the trained model.
- `model.joblib`: The trained model saved after running `train.py`.
- `requirements.txt`: List of packages required to run the scripts.
- `README.md`: This file, providing an overview and instructions for the project.
- `MODELSCARD.md`: Detailed information about the machine learning model.

## Maintainers 👷‍♂️
For any questions or issues, please contact [Danil Zhuravlov](https://www.linkedin.com/in/danil-zhuravlov/).

## License ⚖️
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
