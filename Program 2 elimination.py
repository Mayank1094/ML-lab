import pandas as pd

data = pd.read_csv("trainingdata.csv")
print("training data")
print(data)

Concepts = data.iloc[:, :-1].values
target = data.iloc[:, -1].values

S = Concepts[0].copy()
G = [["?" for i in range(len(S))]]

for i in range(len(Concepts)):
    if target[i] == "yes":
        for j in range(len(S)):
            if S[j] != Concepts[i][j]:
                S[j] = "?"
                G[0][j] = "?"
    else:
        for j in range(len(S)):
            if S[j] != Concepts[i][j]:
                G[0][j] = S[j]
            else:
                G[0][j] = "?"

print("In Specific Hypothesis :")
print(S)

print("In General Hypothesis :")
print(G)
