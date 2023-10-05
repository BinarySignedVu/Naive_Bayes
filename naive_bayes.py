# -------------------------------------------------------------------------
# AUTHOR: Vu Nguyen
# FILENAME: naive_bayes.py
# SPECIFICATION: This program uses Naive Bayes probabilistic approach to compare to other ML/AI algorithms
# FOR: CS 4210- Assignment #2
# TIME SPENT: 4 hrs
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

# importing some Python libraries
import csv
from sklearn.naive_bayes import GaussianNB

feature_map = {
    "Sunny": 1, "Hot": 1, "High": 1, "Strong": 1,
    "Overcast": 2, "Mild": 2, "Normal": 2, "Weak": 2,
    "Rain": 3, "Cool": 3
}

class_map = {
    "Yes": 1,
    "No": 2
}


def read_csv(file_name):
    """Utility function to read CSV and return data without header."""
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        data = list(reader)
    return header, data


def transform_features(data):
    """Convert categorical features to numerical."""
    return [[feature_map[feature] for feature in entry[1:-1]] for entry in data]


def transform_class(data):
    """Convert class labels to numerical."""
    return [class_map[entry[-1]] for entry in data]


header_train, training_data = read_csv("weather_training.csv")
X_train = transform_features(training_data)
Y_train = transform_class(training_data)

# Train the Gaussian Naive Bayes model
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

header_test, test_data = read_csv("weather_test.csv")
X_test = transform_features(test_data)

# Display the header
print(''.join([column.ljust(15) for column in header_test[:-1]]))

# Predict and display results
for entry, features in zip(test_data, X_test):
    prob_yes, prob_no = classifier.predict_proba([features])[0]

    if prob_yes >= 0.75 or prob_no >= 0.75:
        prediction = "Yes" if prob_yes >= 0.75 else "No"
        confidence = prob_yes if prediction == "Yes" else prob_no
        print(''.join([item.ljust(15) for item in entry[:-1]]) + f"{prediction} (confidence): {confidence:.2f}")


