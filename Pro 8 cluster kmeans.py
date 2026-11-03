import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("data.csv")

X = data[['Study Hours', 'Marks']]

kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
data['Cluster'] = kmeans.fit_predict(X)

print(data)

plt.scatter(X['Study Hours'], X['Marks'],
            c=data['Cluster'], cmap='rainbow')

plt.xlabel('Study Hours')
plt.ylabel('Marks')
plt.title('K-Means Clustering')
plt.show()
