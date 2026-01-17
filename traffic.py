import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('TrafficTwoMonth.csv')

'''
situation_counts = df['Traffic Situation'].value_counts()

plt.figure(figsize=(10, 6))
plt.bar(situation_counts.index, situation_counts.values, color='skyblue', edgecolor='black')
plt.title('Distribution of Traffic Situations')
plt.xlabel('Situation')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

'''

def convert_time(time_str):
    time_obj = pd.to_datetime(time_str, format='%I:%M:%S %p')
    return time_obj.hour + time_obj.minute / 60.0

df['Time_Normalized'] = df['Time'].apply(convert_time)


# Convert Day of the Week to numbers (0-6)
day_map = {
    'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 
    'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6
}
df['Day_Encoded'] = df['Day of the week'].map(day_map)


#Map Traffic Situation to a numerical scale
traffic_map = {
    'heavy': 10,
    'high': 7,
    'normal': 4,
    'low': 1
}
df['y_level'] = df['Traffic Situation'].map(traffic_map)

features = ['Time_Normalized', 'Day_Encoded', 'CarCount', 'BusCount', 'TruckCount', 'BikeCount']
X = df[features]
y = df['y_level']

#Train/Test Split (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#200 trees and a depth of 10
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("--- Model Performance ---")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Error: {mean_absolute_error(y_test, y_pred):.2f} (on 1-10 scale)")
print("-------------------------\n")