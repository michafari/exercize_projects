#Import
import pandas as pd
from datetime import datetime

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
print("duplicates.count: ",df.duplicated().sum())

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

#adapt features for model
#OHE for neighbourhoods and room type
#print(df['room_type'].value_counts())
df = pd.get_dummies(df, columns=['neighbourhood'], drop_first=True)
df = pd.get_dummies(df, columns=['room_type'], drop_first=True)

#Pipeline di ML

print(df.head())
print(df.info())
print(df.describe())
'''from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Prepariamo X e y
# Sostituisci 'price' con il nome esatto della tua colonna target
y = df['price']
X = df.drop(columns=['price']) 

# Assicurati di aver tolto eventuali colonne testuali residue (es. nomi, ID)
# X = X.select_dtypes(include=['number']) 

# 2. Dividiamo i dati (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Creiamo il modello Random Forest
# n_estimators è il numero di alberi (100 è un buon punto di partenza)
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# 4. Allenamento
print("Allenamento in corso...")
model.fit(X_train, y_train)

# 5. Predizione e Valutazione
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"--- Risultati ---")
print(f"Errore Medio Assoluto (MAE): {mae:.2f}€")
print(f"Coefficiente R^2 (Precisione): {r2:.2f}")

##RICORDATI DI FARE OOP
'''