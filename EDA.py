import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nptdms import TdmsFile
from sklearn.preprocessing import MinMaxScaler

# Function to load TDMS files from a folder
def load_tdms_files(folder_path):
    data_frames = []
    for file in os.listdir(folder_path):
        if file.endswith(".tdms"):
            tdms_file = TdmsFile.read(os.path.join(folder_path, file))
            for group in tdms_file.groups():
                for channel in group.channels():
                    df = pd.DataFrame({channel.name: channel[:]})
                    df['timestamp'] = channel.time_track()
                    data_frames.append(df)
    return pd.concat(data_frames, axis=1)

# Load data from TDMS files
folder_path = "path/to/your/tdms/files"
data = load_tdms_files(folder_path)

# Basic EDA
print("Data shape:", data.shape)
print("\nData info:")
data.info()

print("\nMissing values:")
print(data.isnull().sum())

# Plot histograms for each feature
data.hist(figsize=(15, 15))
plt.tight_layout()
plt.show()

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(data.select_dtypes(include=[np.number])),
                               columns=data.select_dtypes(include=[np.number]).columns)

# Plot normalized data
normalized_data.plot(figsize=(15, 10))
plt.title("Normalized Time Series Data")
plt.xlabel("Time")
plt.ylabel("Normalized Value")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

# Calculate correlation matrix
correlation_matrix = normalized_data.corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()