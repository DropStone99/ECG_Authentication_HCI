import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load the preprocessed data into a pandas DataFrame
df = pd.read_excel("transpose.xlsx")
df = df.drop(df.columns[[0, 1, 2]], axis=1)

# Remove the string 'Person_' from the last column
df.iloc[: ,-1] = df.iloc[: ,-1].str.replace('Person_', '')

# Convert the 'Person' column to integer type
df.iloc[: ,-1] = df.iloc[: ,-1].astype(int)

X = df.drop(df.columns[[-1]], axis= 1)
y = df.iloc[: ,-1] 
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a GBM model
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)

# Train the GBM model on the training data
gbm.fit(X_train, y_train)

# Use the trained GBM model to make predictions on the testing data
y_pred = gbm.predict(X_test)

# Evaluate the performance of the GBM model using accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', accuracy)