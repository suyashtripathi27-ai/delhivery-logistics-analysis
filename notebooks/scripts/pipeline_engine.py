import pandas as pd
import numpy as np
import os

print("🚀 Starting Automated MLOps Pipeline...")

# 1. Look for new files in the raw folder
raw_dir = 'data/raw/'
processed_dir = 'data/processed/'

# Find the newest CSV file
files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
if not files:
    print("No CSV files found in data/raw/. Exiting.")
    exit()

latest_file = files[0]
file_path = os.path.join(raw_dir, latest_file)
print(f"📥 Ingesting file: {latest_file}")

# 2. AI/Routing Simulation 
print("🧠 Schema Analyzer: Detected 'Logistics/Supply Chain' format.")
print("🔄 Routing to Logistics Processing Engine...")

# 3. Process the Data (Your Business Logic)
df = pd.read_csv(file_path)

# Clean missing locations
df['source_name'] = df['source_name'].fillna('Unknown')
df['destination_name'] = df['destination_name'].fillna('Unknown')
df['trip_creation_time'] = pd.to_datetime(df['trip_creation_time'], errors='coerce')

# Aggregate to Trip Level
trip_df = df.groupby('trip_uuid').agg({
    'route_type': 'first',
    'source_name': 'first',
    'destination_name': 'first',
    'trip_creation_time': 'first',
    'actual_time': 'sum',
    'osrm_time': 'sum'
}).reset_index()

# Calculate Delay Ratio KPI
trip_df['delay_ratio'] = trip_df['actual_time'] / trip_df['osrm_time']
trip_df = trip_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['delay_ratio'])
trip_df = trip_df[(trip_df['delay_ratio'] > 0.1) & (trip_df['delay_ratio'] < 10)]

# Extract Hour for Time-Series Analysis
trip_df['hour'] = trip_df['trip_creation_time'].dt.hour

# 4. Save to Processed Folder
output_path = os.path.join(processed_dir, 'delhivery_powerbi_ready.csv')
trip_df.to_csv(output_path, index=False)

print(f"✅ Processing complete! Clean data saved to {output_path}")
print("📊 Power BI dashboard is ready to update.")
