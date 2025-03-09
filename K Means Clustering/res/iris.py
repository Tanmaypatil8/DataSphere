from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


iris = datasets.load_iris()
X = iris.data
df = pd.DataFrame(X)
df.columns = iris.feature_names
print(df.head())

"""
plt.scatter(df['petal width (cm)'], df['petal length (cm)'], color='blue', marker='o')
plt.title('petal width vs petal length')
plt.xlabel('petal width (cm)')
plt.ylabel('petal length (cm)')
plt.show()
"""

"""
km = KMeans(n_clusters=2)
y_predicted = km.fit_predict((df[['petal width (cm)', 'petal length (cm)']]))
print(y_predicted)

df['cluster'] = y_predicted
print(df.head())
print(km.cluster_centers_)
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]


plt.scatter(df1['petal width (cm)'], df1['petal length (cm)'], color='green')
plt.scatter(df2['petal width (cm)'], df2['petal length (cm)'], color='red')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker='*', label='centroid')
plt.title('petal width vs petal length')
plt.xlabel('petal width (cm)')
plt.ylabel('petal length (cm)')
plt.legend()
plt.show()
"""

# after scaling
scaler = MinMaxScaler()
df[['petal width (cm)', 'petal length (cm)']] = scaler.fit_transform(df[['petal width (cm)', 'petal length (cm)']])
print(df.head())

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['petal width (cm)', 'petal length (cm)']])
print(y_predicted)

df['cluster'] = y_predicted
print(df.head())

df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]


plt.scatter(df1['petal width (cm)'], df1['petal length (cm)'], color='green')
plt.scatter(df2['petal width (cm)'], df2['petal length (cm)'], color='red')
plt.scatter(df3['petal width (cm)'], df3['petal length (cm)'], color='black')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker='*', label='centroid')
plt.title('petal width vs petal length')
plt.xlabel('petal width (cm)')
plt.ylabel('petal length (cm)')
plt.legend()
plt.show()


sse = []
k_rng = range(1, 10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['petal width (cm)', 'petal length (cm)']])
    sse.append(km.inertia_)
    # print(km.inertia_)
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng, sse)
plt.show()

