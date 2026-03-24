#Import
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

#Visualization console options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 300)


def read_csv(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    #drop all null column
    df.drop(["neighbourhood_group"], axis=1, inplace=True)
    # drop data without price
    df = df.dropna(subset=['price'])
    # drop unuseful columns
    df.drop(["id"], axis=1, inplace=True)
    df.drop(["name"], axis=1, inplace=True)
    df.drop(["host_id"], axis=1, inplace=True)
    df.drop(["host_name"], axis=1, inplace=True)
    df.drop(["license"], axis=1, inplace=True)
    return df

def feature_eng(df):
    # at least one recension in last 12 months to rely on actual price
    df = df[df['number_of_reviews_ltm'] >= 3]
    # remove reviews_per_month because redundant after last year filter
    df.drop(["reviews_per_month"], axis=1, inplace=True)

    # transform last review data into days to last review, create new feature
    today = "2025-09-15"
    date_list = df["last_review"]
    date_format = "%Y-%m-%d"
    today_to_date = datetime.strptime(today, date_format)
    days_diff = []
    for d in date_list:
        d_to_date = datetime.strptime(d, date_format)
        diff = (today_to_date - d_to_date).days
        days_diff.append(diff)
    df["last_review_days"] = days_diff
    df.drop(["last_review"], axis=1, inplace=True)

    # drop rows with too high price
    df = df[(df["price"] <= 250) & (df["price"] >= 50)]
    # drop hotel room e shared room
    df = df[(df["room_type"] == "Private room") | (df["room_type"] == "Entire home/apt")]

    # target encoding for neighbourhoods + room_type
    df['neighborhood_room'] = df['neighbourhood'].astype(str) + "_" + df['room_type'].astype(str)
    combined_means = df.groupby('neighborhood_room')['price'].mean()
    df['neigh_room_feature'] = df['neighborhood_room'].map(combined_means)
    df.drop(['neighbourhood'], axis=1, inplace=True)
    df.drop(['room_type'], axis=1, inplace=True)
    df.drop(["neighborhood_room"], axis=1, inplace=True)

    ##test without columns
    #df.drop(["latitude", "longitude"], axis=1, inplace=True)
    #df.drop(["availability_365", "minimum_nights", "number_of_reviews", "calculated_host_listings_count", "number_of_reviews_ltm", "last_review_days"], axis=1, inplace=True)

    return df

def pipeline(df):
    y = df['price']
    X = df.drop(columns=['price'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return y_test, predictions, model, X


def test_analysis(y_test, predictions, model, X):
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    print(f"--- test analysis ---")
    print(f"Coefficient R^2 (Precision): {r2:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}€")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    # Features importance
    importances = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    print("\n--- Feature Importance  ---")
    print(feature_importance_df)

    return

def visualize(y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Real Price')
    plt.ylabel('Predicted Price')
    plt.title('Real vs Predicted')
    plt.show()
    return


def main():
    #import file cvs and creating dataframe
    dataframe = read_csv('listings.csv')
    #make a copy
    df_original = dataframe.copy()
    #clean dataframe
    cleaned_dataframe = clean_data(dataframe)
    #features engineering
    feat_dataframe = feature_eng(cleaned_dataframe)
    print(feat_dataframe.head())
    print(feat_dataframe.describe())
    #training model output
    test_prices, predicted_prices, model, Features = pipeline(feat_dataframe)
    #analysis metrics
    test_analysis(test_prices, predicted_prices, model, Features)
    #plot
    visualize(test_prices, predicted_prices)

    return

if __name__ == "__main__": #in this way if I import this file in other project it doesn't run everything inside main
    main()

