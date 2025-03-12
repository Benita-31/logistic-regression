import pandas as pd  # Importing pandas for data manipulation
from sklearn.model_selection import train_test_split  # To split data into training and testing sets
from sklearn.linear_model import LogisticRegression  # Importing Logistic Regression model
from sklearn.metrics import accuracy_score  # To evaluate the model's performance

# Load the dataset with Windows-1252 encoding (commonly referred to as ANSI)
df = pd.read_csv('data.csv', encoding='windows-1252')

# Selecting the relevant columns: 'Age' and 'BMI' as features, 'Risk' as the target
X = df[['Age', 'BMI (kg/m²)']]  # Make sure these columns exist in the dataset
y = df['Risk']  # 'Risk' column should exist and be properly labeled

# Splitting the dataset into 70% training and 30% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating an instance of the Logistic Regression model
model = LogisticRegression()

# Training the model using the training data
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Calculating and displaying the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100}%')  # Converting to percentage format for readability

# Creating a small test case with an example person (Age 40, BMI 31) to see the prediction
new_data = pd.DataFrame([[40, 31]], columns=['Age', 'BMI (kg/m²)'])  # Column names should match training data

# Using the trained model to predict the risk category for the new input
prediction = model.predict(new_data)

# Printing the result in a readable format
print(f'Prediction for age 40 and BMI 31: {"High Risk" if prediction[0] == 1 else "Low Risk"}')
