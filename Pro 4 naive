import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = pd.read_csv("tennis.csv")
encoder = LabelEncoder()

for column in data.columns:
    data[column] = encoder.fit_transform(data[column])

X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

model = GaussianNB()
model.fit(X, Y)

prediction = model.predict(X)

print("Actual Output:")
print(list(Y))

print("\nPredicted Output:")
print([int(i) for i in prediction])

print("\nAccuracy:", accuracy_score(Y, prediction) * 100, "%")
