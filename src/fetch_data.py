from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

def fetch_data():
    # fetch dataset 
    firm_bankruptcy = fetch_ucirepo(id=572) 
      
    # data (as pandas dataframes) 
    X = firm_bankruptcy.data.features 
    y = firm_bankruptcy.data.targets 
      
    # Combine features and target
    df = pd.concat([X, y], axis=1)
    
    # Save to CSV
    os.makedirs('firm_bankruptcy_prediction/data', exist_ok=True)
    output_path = 'firm_bankruptcy_prediction/data/data.csv'
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    fetch_data()
