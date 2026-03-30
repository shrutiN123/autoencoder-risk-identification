import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = pd.read_csv("data/sample_dataset.csv")

# Drop non-numeric column
if 'risk_type' in data.columns:
    data_numeric = data.drop(columns=['risk_type'])
else:
    data_numeric = data

# -----------------------------
# 2. Normalize Data
# -----------------------------
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_numeric)

# -----------------------------
# 3. Build Autoencoder
# -----------------------------
input_dim = data_scaled.shape[1]

input_layer = Input(shape=(input_dim,))
encoder = Dense(8, activation="relu")(input_layer)
encoder = Dense(4, activation="relu")(encoder)

decoder = Dense(8, activation="relu")(encoder)
decoder = Dense(input_dim, activation="sigmoid")(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer="adam", loss="mse")

# -----------------------------
# 4. Train Model
# -----------------------------
history = autoencoder.fit(
    data_scaled,
    data_scaled,
    epochs=50,
    batch_size=4,
    validation_split=0.2,
    verbose=1
)

# -----------------------------
# 5. Reconstruction Error
# -----------------------------
reconstructions = autoencoder.predict(data_scaled)
mse = np.mean(np.power(data_scaled - reconstructions, 2), axis=1)

# Threshold (you can tune this)
threshold = np.percentile(mse, 90)

# Flag anomalies
anomalies = mse > threshold

# -----------------------------
# 6. Results
# -----------------------------
results = data.copy()
results['Reconstruction_Error'] = mse
results['Anomaly'] = anomalies

print("\n🔍 Detected Risks:")
print(results[results['Anomaly'] == True])

# -----------------------------
# 7. Visualization
# -----------------------------
plt.figure()
plt.plot(mse, label="Reconstruction Error")
plt.axhline(threshold, color='r', linestyle='--', label="Threshold")
plt.title("Anomaly Detection using Autoencoder")
plt.xlabel("Data Point Index")
plt.ylabel("Error")
plt.legend()

# Save graph
plt.savefig("results/output.png")
plt.show()
