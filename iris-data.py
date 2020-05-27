import os
import sklearn
import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing
from sklearn.neighbors import KNeighborsClassifier

data_sets = os.path.join(os.path.expanduser('~'), 'Desktop/Datasets')
data = pd.read_csv(f"{data_sets}/iris.csv")

proc = preprocessing.LabelEncoder()
sepal_length = proc.fit_transform(list(data["sepal.length"]))
sepal_width = proc.fit_transform(list(data["sepal.width"]))
petal_length = proc.fit_transform(list(data["petal.length"]))
petal_width = proc.fit_transform(list(data["petal.width"]))
variety = proc.fit_transform(list(data["variety"]))

predict = "variety"

x = list(zip(sepal_length, sepal_width, petal_length, petal_width))
y = list(variety)


vari = ["Setosa", "Virginica", "Versicolor"]
best = 0
worst = 100
for i in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.9)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    if accuracy > best:
        best = accuracy

    elif accuracy < worst:
        worst = accuracy


    prediction = model.predict(x_test)

    print(f"Prediction:\t{vari[prediction[i]].ljust(10)}\tActual: {vari[y_test[i]].ljust(10)}\tAccuracy: {str(round(accuracy * 100, 2)).ljust(5)}%\tData: {x_test[i]}")

print(f"\nHighest Accuracy: {round((best * 100), 2)}%")
print(f"Worst Accuracy: {round((worst * 100), 2)}%")

