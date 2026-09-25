import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mydata = pd.read_csv("dataset.csv")

# A
print("Display first 5 records:")
print(mydata.head())

print("\nDisplay last 5 records:")
print(mydata.tail())

print("\nShape:")
print(mydata.shape)

print("\nColumn Information:")
mydata.info()

print("\nStatistical Summary:")
print(mydata.describe())

# B
print("\nMissing Values Per Column:")
print(mydata.isnull().sum())

print("\nDuplicate Records:")
m = mydata.drop_duplicates()
print(m.shape)

print("\nCategorical Attributes:")
print(mydata.select_dtypes(include=['object', 'category']).columns)

print("\nNumerical Attributes:")
print(mydata.select_dtypes(include=['int64', 'float64']).columns)

print("\nUnique Values:")
print(mydata.nunique())

# C
num_cols = mydata.select_dtypes(include='number').columns

print("\nMean:")
print(mydata[num_cols].mean())

print("\nMedian:")
print(mydata[num_cols].median())

print("\nMode:")
print(mydata[num_cols].mode())

print("\nStandard Deviation:")
print(mydata[num_cols].std())

print("\nMinimum Values:")
print(mydata[num_cols].min())

print("\nMaximum Values:")
print(mydata[num_cols].max())

variance = mydata[num_cols].var()

print("\nAttribute with Highest Variance:")
print(variance)

# D
plt.hist(mydata['Study_Hours'])
plt.title("Study Hours Distribution")
plt.show()

plt.hist(mydata['Attendence'])    # Use correct column name
plt.title("Attendance Distribution")
plt.show()

sns.boxplot(data=mydata[['Study_Hours', 'Internal_Marks']])
plt.show()

sns.scatterplot(x='Attendence', y='Study_Hours', data=mydata)
plt.show()

corr = mydata.corr(numeric_only=True)

print("\nCorrelation Matrix:")
print(corr)
