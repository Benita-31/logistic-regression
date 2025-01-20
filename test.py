import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset with Windows-1252 encoding (commonly referred to as ANSI)
df = pd.read_csv('data.csv', encoding='windows-1252')

# Split the data into features (X) and target (y)
X = df[['Age', 'BMI (kg/m²)']]  # Make sure these columns exist in the CSV file
y = df['Risk']  # Ensure the 'Risk' column exists

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100}%')

# Example of predicting the outcome for an unknown input (e.g., age=40, BMI=31)
new_data = pd.DataFrame([[40, 31]], columns=['Age', 'BMI (kg/m²)'])  # Ensure the column names match
prediction = model.predict(new_data)
print(f'Prediction for age 40 and BMI 31: {"High Risk" if prediction[0] == 1 else "Low Risk"}')
