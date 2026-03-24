Sprawozdanie: Klasyfikacja kwiatów Iris
Indeks: 119066
Metoda: Algorytm KNN (K-Nearest Neighbors)

1. O danych
W projekcie wykorzystałem zbiór 150 próbek irysów. Każdy kwiat opisany jest przez cztery cechy: długość i szerokość płatka oraz działki kielicha. Dane dzielą się na trzy gatunki: Setosa, Versicolour i Virginica.

2. Model i badanie
Wybrałem algorytm KNN z parametrem K=3. Program uczył się na 80% danych, a pozostałe 20% służyło do sprawdzenia, czy potrafi on poprawnie rozpoznać gatunek.

3. Wyniki i wykres
Model uzyskał 100% skuteczności (wynik 1.00 we wszystkich metrykach[accuracy, precision, recall, f1-score]). Na wykresie widać, że czerwona grupa (Setosa) jest wyraźnie oddzielona, co ułatwiło zadanie komputerowi. Pozostałe dwie grupy są bliżej siebie, ale algorytm i tak bezbłędnie je rozróżnił.

Wnioski z projektu
Skuteczność algorytmu: Algorytm KNN świetnie radzi sobie ze zbiorami danych takimi jak Iris, osiągnął on maksymalną precyzję.


Wizualizacja: Dzięki technice t-SNE mogliśmy zobaczyć 4-wymiarowe dane na płaszczyźnie, co pozwoliło zrozumieć, jak działał model.

Podział danych: Zastosowanie podziału na zbiór treningowy i testowy pozwoliło ocenić, że model faktycznie potrafi klasyfikować nowe dane, a nie tylko te, które już widział.

Narzędzia: Biblioteki takie jak scikit-learn i pandas znacznie upraszczają proces budowania modeli machine learningowych nawet dla początkujących.
