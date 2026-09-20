import pandas as pd

data = pd.read_csv("trainingdata.csv")
print("trainingdata")
print(data)

Concept = data.iloc[:, :-1].values
target = data.iloc[:, -1].values

hypothesis = Concept[0].copy()

for i in range(len(target)):
    if target[i] == "yes":
        for j in range(len(hypothesis)):
            if hypothesis[j] != Concept[i][j]:
                hypothesis[j] = "?"

print("In final hypothesis")
print(hypothesis)
