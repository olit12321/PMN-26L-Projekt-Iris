import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE

# 1. Wczytanie danych
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv("bezdekIris.data", names=columns)
X = df.drop(columns=["class"])
y = df["class"]

# 2. Krótka analiza danych (Wykresy Pairplot)
plt.figure()
sns.pairplot(df, hue="class", palette="Set1", markers=["o", "s", "D"])
plt.savefig("pairplot_iris.png")

# 3. Klasyfikacja KNN (Grupa 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Przewidywania dla zbioru testowego (do metryk)
y_pred_test = knn.predict(X_test)

# Obliczenie metryk
acc = accuracy_score(y_test, y_pred_test)
prec = precision_score(y_test, y_pred_test, average='macro')
rec = recall_score(y_test, y_pred_test, average='macro')
f1 = f1_score(y_test, y_pred_test, average='macro')

print(f"Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")
print(f"F1-score: {f1:.2f}")

# 4. Wizualizacja t-SNE po klasyfikacji
# Redukujemy wymiary całego zbioru do 2D
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Przewidujemy klasy dla CAŁEGO zbioru, aby pokazać działanie modelu na t-SNE
y_pred_all = knn.predict(X)

# Mapowanie klas na numery dla kolorów i kształtów
class_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y_num_true = y.map(class_mapping)
y_num_pred = pd.Series(y_pred_all).map(class_mapping)

plt.figure(figsize=(10, 7))
markers = ['o', 's', '^']

for species, marker in zip(class_mapping.keys(), markers):
    # Maski dla prawdziwych klas
    mask = (y == species)
    # Kolor punktu to przewidywanie KNN (y_num_pred), kształt to prawdziwa klasa
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                c=y_num_pred[mask], cmap='viridis', 
                marker=marker, s=100, edgecolor='k', label=f'Prawdziwa: {species}')

plt.title('t-SNE: Kolory = Przewidywania KNN, Kształty = Prawdziwe Klasy')
plt.xlabel('Wymiar 1')
plt.ylabel('Wymiar 2')
plt.legend()
plt.savefig("tsne_knn_iris.png")