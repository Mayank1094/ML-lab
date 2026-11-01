from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
pred = model.predict(X_test)

correct = wrong = 0

for a, p in zip(y_test, pred):
    if a == p:
        print("Correct:", iris.target_names[p])
        correct += 1
    else:
        print("Wrong: Actual =", iris.target_names[a], "Predicted =", iris.target_names[p])
        wrong += 1

print("Total Correct:", correct)
print("Total Wrong:", wrong)
print("Accuracy:", accuracy_score(y_test, pred) * 100, "%")
