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
pd.set_option('display.width', 150)

#Import file cvs and dataset analysis (by creating dataframe)
df = pd.read_csv('listings.csv')

#initial dataset cleaning
df_original = df.copy()
df.drop(["neighbourhood_group"], axis=1, inplace=True)

#Initial features analysis
#notes
##data cleaning

#at least one recension in last 12 months to rely on actual price
df = df[df['number_of_reviews_ltm']>=1]
#drop data without price
df = df.dropna(subset=['price'])
#drop unuseful columns
df.drop(["id"], axis=1, inplace=True)
df.drop(["name"], axis=1, inplace=True)
df.drop(["host_id"], axis=1, inplace=True)
df.drop(["host_name"], axis=1, inplace=True)
df.drop(["license"], axis=1, inplace=True)

#remove duplicates if exists
#print("duplicates.count: ",df.duplicated().sum())

#remove reviews_per_month because redundant after last year filter
df.drop(["reviews_per_month"], axis=1, inplace=True)
#transform last review data into days to last review, create new feature
today = "2025-09-15"
date_list = df["last_review"]

date_format = "%Y-%m-%d"
today_to_date = datetime.strptime(today, date_format)

days_diff = []
for d in date_list:
    d_to_date = datetime.strptime(d, date_format)
    diff = (today_to_date-d_to_date).days
    days_diff.append(diff)

df["last_review_days"] = days_diff
df.drop(["last_review"], axis=1, inplace=True)

#drop rows with too high price
df = df[df["price"]<=600]

#adapt features for model
#OHE for room type
#print(df['room_type'].value_counts())
df = pd.get_dummies(df, columns=['room_type'], drop_first=True)

#target encoding for neighbourhoods
mean_prices = df.groupby('neighbourhood')['price'].mean()
df['neighbourhood_avg_price'] = df['neighbourhood'].map(mean_prices)
print(df.head())

df.drop(['neighbourhood'], axis=1, inplace=True)

'''
print(df.head())
print(df.info())
print(df.describe())
'''

##test without availability
df.drop(["availability_365", "last_review_days", "number_of_reviews"], axis=1, inplace=True)

#Pipeline di ML

y = df['price']
X = df.drop(columns=['price'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42)

print("Training in progress...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
#metrics
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

print(f"--- Results ---")
print(f"Coefficient R^2 (Precision): {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}€")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

##RICORDATI DI FARE OOP

#Features importance
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print("\n--- Feature Importance  ---")
print(feature_importance_df)

###
print("\n--- Stats Real Price vs Predicted ---")
print(f"Mean Real Price: {y_test.mean():.2f}€")
print(f"Mean Predicted Price: {predictions.mean():.2f}€")
print(f"Max Real Price: {y_test.max():.2f}€")
print(f"Max Predicted Price: {predictions.max():.2f}€")

##
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Real Price')
plt.ylabel('Predicted Price')
plt.title('Real vs Predicted')
plt.show()


#Prova a togliere shared rooms a hotel rooms
