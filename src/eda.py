import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda():
    df = pd.read_csv('firm_bankruptcy_prediction/data/data.csv')
    
    os.makedirs('firm_bankruptcy_prediction/eda_output', exist_ok=True)
    
    # Basic Info
    with open('firm_bankruptcy_prediction/eda_output/info.txt', 'w') as f:
        f.write("Shape:\n")
        f.write(str(df.shape) + "\n\n")
        f.write("Class Distribution:\n")
        f.write(str(df['Bankrupt?'].value_counts(normalize=True)) + "\n\n")
        f.write("Missing Values:\n")
        f.write(str(df.isnull().sum().sum()) + "\n")

    # Class Balance Plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Bankrupt?', data=df)
    plt.title('Class Distribution')
    plt.savefig('firm_bankruptcy_prediction/eda_output/class_distribution.png')
    plt.close()

    print("EDA completed. Results saved to firm_bankruptcy_prediction/eda_output/")

if __name__ == "__main__":
    perform_eda()
