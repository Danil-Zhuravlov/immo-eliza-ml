import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# opening the data of properties
path = "data/properties.csv"
df = pd.read_csv(path)

missing_values = df.isnull().sum()
numerical_stats = df.describe()

class Preprocessing:

    @staticmethod
    def fill_nan_with_median(df, column_name):
        """
        Fill missing values in the specified column(s) with the median value.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        column_name (str or list): The name(s) of the column(s) to fill.

        Returns:
        None
        """
        if isinstance(column_name, list):
            for column in column_name:
                median_of_column = df[column].median()
                df.fillna({column: median_of_column}, inplace=True)
        else:
            median_of_column = df[column_name].median()
            df.fillna({column_name: median_of_column}, inplace=True)

        return None


    @staticmethod
    def fill_nan_with_0(df, column_name):
        """
        Fill NaN values in the specified column(s) with 0.

        Parameters:
        - df (pandas.DataFrame): The DataFrame containing the data.
        - column_name (str or list): The name(s) of the column(s) to fill NaN values in.

        Returns:
        None
        """
        if isinstance(column_name, list):
            for column in column_name:
                df.fillna({column: 0}, inplace=True)
        else:
            df.fillna({column_name: 0}, inplace=True)

        return None
    
    @staticmethod
    def iqr(dataframe, column_names):
        filtered_df = dataframe.copy()

        if isinstance(column_names, list):
            for column_name in column_names:
                Q1 = dataframe[column_name].quantile(0.25)
                Q3 = dataframe[column_name].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                filtered_df = filtered_df[(filtered_df[column_name] >= lower_bound) & (filtered_df[column_name] <= upper_bound)]
        else:
            Q1 = dataframe[column_names].quantile(0.25)
            Q3 = dataframe[column_names].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            filtered_df = filtered_df[(filtered_df[column_names] >= lower_bound) & (filtered_df[column_names] <= upper_bound)]

        return filtered_df

prep = Preprocessing

prep.fill_nan_with_median(df, 'total_area_sqm')
prep.fill_nan_with_0(df, ['surface_land_sqm', 'terrace_sqm', 'garden_sqm'])

selected_features = df[['price', 'total_area_sqm', 'nbr_bedrooms', 'garden_sqm', 'fl_double_glazing']]

filtered_selected_features = prep.iqr(selected_features, ['price', 'total_area_sqm', 'nbr_bedrooms', 'garden_sqm'])

# target
y = filtered_selected_features['price'].values

# features
X = filtered_selected_features.drop(columns=['price']).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

score = model.score(X_test_scaled, y_test)

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

print(f'Error: €{round(rmse)}')

print(f'Model\'s score: {score}')
