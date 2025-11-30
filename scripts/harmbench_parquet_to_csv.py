import pandas as pd

for folder in ['contextual', 'copyright', 'standard']:
    # Read the parquet file
    parquet_file = f'/home/ubuntu/AutoDAN-custom/data/HarmBench/{folder}/train-00000-of-00001.parquet'
    df = pd.read_parquet(parquet_file)

    # Convert to CSV
    csv_file = f'/home/ubuntu/AutoDAN-custom/data/HarmBench/{folder}/train.csv'
    df.to_csv(csv_file, index=False)

    print(f"Converted {parquet_file} to {csv_file}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())