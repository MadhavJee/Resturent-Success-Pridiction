#Importing All Important Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
df = pd.read_csv(R"D:\MY CLASS\Python datasets\Zomato Data\zomato.csv")

#checking for the data
df.head()
df.info()
df.describe()

##Data Cleaning

# Cleaning the 'rate' column
def clean_rate(rate):
    if pd.isna(rate) or rate == 'NEW':
        return np.nan
    try:
        return float(rate.split('/')[0])
    except:
        return np.nan

df['rating'] = df['rate'].apply(clean_rate)

# Defineing success (1 if rating >= 4.0, else 0)
df['success'] = np.where(df['rating'] >= 4.0, 1, 0)

# Handleing missing values by dropping rows with no target
df = df.dropna(subset=['success'])

# Cleaning 'approx_cost(for two people)' by removing commas and converting to numeric
df['cost'] = pd.to_numeric(df['approx_cost(for two people)'].str.replace(',', ''), errors='coerce')

# Cleaning 'votes' by converting to numeric
df['votes'] = pd.to_numeric(df['votes'], errors='coerce')

# Droping rows where 'cost' or 'votes' are NaN
df = df.dropna(subset=['cost', 'votes'])
