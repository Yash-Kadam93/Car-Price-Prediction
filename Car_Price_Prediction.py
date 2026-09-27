# Import Required Libraries
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import warnings
warnings.filterwarnings('ignore')

# Load Dataset
df=pd.read_csv('ford.csv')
print(df)

#EDA
print(df.shape)
print(df.info())
print(df.head())
print(df.describe())
print(df.isnull().sum())

# Data Visualization 
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.show()

columns=['model','year','transmission','fuelType','tax','mpg','engineSize']
for col in columns:
    plt.figure(figsize=(8,6))
    sns.boxplot(data=df,x=col,y='price')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

sns.scatterplot(data=df,x='mileage',y='price')
plt.show()


# Feature and Target Separation
x=df.drop(columns=['price'])
y=df['price']
print(x)
print(y)


# Encoding Categorical Features
x=pd.get_dummies(x,columns=['model','transmission','fuelType'],drop_first=True)
print(x)
x=x.astype(int)
print(x)

# Feature Scaling
numeric_cols=['year','mileage','tax','mpg','engineSize']
scaler=StandardScaler()
x[numeric_cols]=scaler.fit_transform(x[numeric_cols])
print(x)


# Split the Dataset, Create and Train the Model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
print(y_test)


# Model Evaluation
r2=r2_score(y_test,y_pred)

n=x_test.shape[0]
p=x_test.shape[1]
adjusted_r2=1-((1-r2)*(n-1)/(n-p-1))


print("R2 Score:",r2)
print("Adjudted R2 Score:",adjusted_r2)
