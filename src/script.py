# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Import data with your folder path  
df = pd.read_csv('', sep=';', decimal=',')

# Select features
features = df[['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']]

# Clean data 
features = features.replace(-200, np.nan)
features = features.dropna()
print(features.shape)
print(features.head())

# Isolation Forest hyperparameters 
n_estimators = 100
contamination = 0.01
sample_size = 256
random_state = 42

# -------- Train Isolation Forest --------
iso_forest = IsolationForest(
    n_estimators=n_estimators,
    contamination=contamination,
    max_samples=sample_size,
    random_state=random_state
)
iso_forest.fit(features)

# Calculate anomaly scores and classify anomalies
data = df.loc[features.index].copy()
data['anomaly_score'] = iso_forest.decision_function(features)
data['anomaly'] = iso_forest.predict(features)
data['anomaly'].value_counts()

# Visualization of the results
plt.figure(figsize=(10, 5))

# Plot normal instances
normal = data[data['anomaly'] == 1]
plt.scatter(normal.index, normal['anomaly_score'], label='Normal')

# Plot anomalies
anomalies = data[data['anomaly'] == -1]
plt.scatter(anomalies.index, anomalies['anomaly_score'], label='Anomaly')
plt.xlabel('Instance')
plt.ylabel('Anomaly Score')
plt.legend()
plt.show()

# Visualization of the results
plt.figure(figsize=(5, 5))

# Plot non-anomalies then anomalies
plt.scatter(normal['CO(GT)'], normal['NO2(GT)'], label='Normal')
plt.scatter(anomalies['CO(GT)'], anomalies['NO2(GT)'], label='Anomaly')
plt.xlabel('CO(GT)')
plt.ylabel('NO2(GT)')
plt.legend()
plt.show()

# -------- Attach anomaly scores and flags --------
df_anomalies = df.loc[features.index].copy()
df_anomalies['anomaly_score'] = iso_forest.decision_function(features)
df_anomalies['is_anomaly'] = (iso_forest.predict(features) == -1).astype(int)

# Export output with your folder path
df_anomalies.to_csv('', index=False)
