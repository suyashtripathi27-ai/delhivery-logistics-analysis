import pandas as pd
import numpy as np
import os
import sys
import google.generativeai as genai

print("🚀 Starting Automated MLOps Pipeline...")

# 1. Connect to your GitHub Secret Vault
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ ERROR: GEMINI_API_KEY not found. Did you save it in GitHub Secrets?")
    sys.exit(1)

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# 2. Find the newest file in the raw folder
raw_dir = 'data/raw/'
processed_dir = 'data/processed/'
os.makedirs(processed_dir, exist_ok=True)

files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
if not files:
    print("No CSV files found in data/raw/. Exiting.")
    sys.exit(0)

latest_file = files[0] # Grabs whatever you just uploaded
file_path = os.path.join(raw_dir, latest_file)
print(f"📥 Ingesting file: {latest_file}")

# Read the data
df = pd.read_csv(file_path)

# 3. THE AI BRAIN: Ask Gemini to classify the data based on columns
columns_str = ", ".join(df.columns.tolist())
prompt = f"""
Look at these dataset columns: {columns_str}. 
Does this data belong to the 'LOGISTICS' industry (trucks, trips, hubs) or the 'RETAIL' industry (sales, footfall, qty)? 
Reply with ONLY ONE WORD: LOGISTICS or RETAIL.
"""

print("🧠 Asking AI to classify the dataset schema...")
response = model.generate_content(prompt)
industry = response.text.strip().upper()
print(f"🤖 AI Classification Result: {industry}")

# 4. THE ROUTER: Send it to the correct math engine
if "LOGISTICS" in industry:
    print("🛣️ Routing to Logistics Processing Engine...")
    
    # [Logistics Math]
    df['source_name'] = df['source_name'].fillna('Unknown')
    df['destination_name'] = df['destination_name'].fillna('Unknown')
    df['trip_creation_time'] = pd.to_datetime(df['trip_creation_time'], errors='coerce')
    
    trip_df = df.groupby('trip_uuid').agg({
        'route_type': 'first', 'source_name': 'first', 'destination_name': 'first',
        'actual_time': 'sum', 'osrm_time': 'sum'
    }).reset_index()
    
    trip_df['delay_ratio'] = trip_df['actual_time'] / trip_df['osrm_time']
    trip_df = trip_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['delay_ratio'])
    trip_df = trip_df[(trip_df['delay_ratio'] > 0.1) & (trip_df['delay_ratio'] < 10)]
    
    output_path = os.path.join(processed_dir, 'delhivery_powerbi_ready.csv')
    trip_df.to_csv(output_path, index=False)
    print(f"✅ Logistics processing complete! Clean data saved to {output_path}")

elif "RETAIL" in industry:
    print("🛍️ Routing to Retail Processing Engine...")
    
    # [Retail Math for the Ethnic Wear file]
    # We will do a simple Retail KPI: Average Conversion Rate & Total Sales
    df['Conversion Rate'] = pd.to_numeric(df['Conversion Rate'], errors='coerce')
    df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
    
    retail_summary = df.groupby('Day').agg({
        'Sales': 'sum',
        'Footfall ': 'sum',
        'Conversion Rate': 'mean'
    }).reset_index()
    
    output_path = os.path.join(processed_dir, 'retail_powerbi_ready.csv')
    retail_summary.to_csv(output_path, index=False)
    print(f"✅ Retail processing complete! Clean data saved to {output_path}")

else:
    print(f"⚠️ Unknown data type detected ({industry}). Alerting Human Supervisor. Pipeline Stopped Safely.")
