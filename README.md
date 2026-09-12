# Customer Churn Analysis using Python

```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
```

# 1. Load Dataset
```
df = pd.read_csv("Customer Churn.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)
```

# 2. Data Cleaning
```
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

df = df.drop_duplicates()
```

# 3. Churn Analysis
```
print("\nChurn Count:")
print(df["Churn"].value_counts())

print("\nChurn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)
```

# 4. Churn Visualization
```
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.show()
```

# 5. Churn by Contract
```
contract = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn by Contract:")
print(contract.round(2))

contract.plot(kind="bar")
plt.title("Churn by Contract Type")
plt.ylabel("Churn Percentage")
plt.xticks(rotation=0)
plt.show()
```

# 6. Tenure vs Churn
```
sns.boxplot(
    data=df,
    x="Churn",
    y="tenure"
)

plt.title("Tenure vs Churn")
plt.show()
```

# 7. Monthly Charges vs Churn
```
sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)

plt.title("Monthly Charges vs Churn")
plt.show()
```

# 8. Prepare Data for Machine Learning
```
data = df.drop("customerID", axis=1)

data = pd.get_dummies(
    data,
    drop_first=True
)

X = data.drop("Churn_Yes", axis=1)
y = data["Churn_Yes"]
```

# 9. Final Business Insights
```
print("\nBusiness Insights:")
print("1. Month-to-month contract customers have higher churn.")
print("2. Customers with shorter tenure are more likely to churn.")
print("3. Higher monthly charges are associated with higher churn.")
print("4. Businesses can target high-risk customers with retention offers.")
```

## About the Project

This project analyzes customer churn using Python and identifies the factors that influence customers to leave a company.

## Tools Used

* Python
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

## Analysis Performed

* Data cleaning
* Churn percentage analysis
* Churn by contract type
* Tenure analysis
* Monthly charges analysis
* Data visualization
* Customer churn prediction using Logistic Regression

## Key Insights

* Month-to-month customers have higher churn.
* Customers with shorter tenure are more likely to leave.
* Higher monthly charges are associated with increased churn.
* Churn prediction can help businesses target customers who may leave.

Thank You
