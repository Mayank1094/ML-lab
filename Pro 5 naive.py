from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

documents = [
    "India won the cricket match",
    "The football team played well",
    "The election results where announced",
    "The government passed a new law"
]

labels = ["Sports", "Sports", "Politics", "Politics"]

cv = CountVectorizer()
X = cv.fit_transform(documents)

model = MultinomialNB()
model.fit(X, labels)

test_docs = [
    "India won the cricket match",
    "The government announced a new election"
]

actual = ["Sports", "Politics"]

X_test = cv.transform(test_docs)
predicted = model.predict(X_test)

print("Actual labels:", actual)
print("Predicted labels:", list(predicted))
print("Accuracy:", accuracy_score(actual, predicted))
print("Precision:", precision_score(actual, predicted, pos_label="Sports"))
print("Recall:", recall_score(actual, predicted, pos_label="Sports"))
