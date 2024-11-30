import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv(r"C:\Users\Admin\Downloads\Batting.csv")

data['out/not_out'] = data['out/not_out'].apply(lambda x: 1 if x == 'out' else 0)

data = data.dropna()

X = data[['runs', 'balls', '4s']]
y = data['out/not_out']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Not Out", "Out"])

print("=== KNN Classification Results ===")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)

print("\nSample Predictions (Test Data):")
test_results = pd.DataFrame(X_test, columns=['runs', 'balls', '4s'])
test_results['Actual'] = y_test.values
test_results['Predicted'] = y_pred
print(test_results.head(10))
