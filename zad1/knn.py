import pandas as pd

df = pd.read_csv("bezdekIris.data",names=["sepal_length","sepal_width","petal_length","petal_width","class"])
print(df.head())
X=df.drop(columns=["class"])
y=df["class"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.neighbors import KNeighborsClassifier


knn = KNeighborsClassifier(n_neighbors=3)


knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")
print(f"F1-score: {f1:.2f}")


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)


plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='Set1')
plt.title('Wizualizacja zbioru Iris za pomocą t-SNE')
plt.xlabel('Wymiar 1')
plt.ylabel('Wymiar 2')

plt.savefig('tsne_iris.png')