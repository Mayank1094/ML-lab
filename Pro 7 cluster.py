import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")

model = GaussianMixture(n_components=3, random_state=0)
model.fit(data)

data["Cluster"] = model.predict(data)

print(data)

plt.scatter(data["Marks"], data["Attendence"], c=data["Cluster"])
plt.xlabel("Marks")
plt.ylabel("Attendence")
plt.title("EM Algorithm Clustering")
plt.show()
