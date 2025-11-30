import pandas as pd

data = pd.read_csv("/home/ubuntu/AutoDAN-custom/data/HarmBench/contextual/train.csv")

#uniue of category
unique_categories = data['category'].unique()

print("Unique categories in the dataset:", unique_categories)