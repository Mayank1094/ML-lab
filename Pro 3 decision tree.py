import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("play-tennis.csv")
print("Training Dataset :\n")
print(data)

encoder = LabelEncoder()

for column in data.columns:
    data[column] = encoder.fit_transform(data[column])

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, Y)

sample = [[2, 1, 0, 1]]
result = model.predict(sample)

print("\n prediction :")

if result[0] == 1:
    print("play tennis = Yes")
else:
    print("play tennis = No")
