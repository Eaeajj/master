import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Не нормализованные данные
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Признак
y = np.array([2, 4, 5, 4, 5])  # Целевая переменная

# Обучение модели линейной регрессии без нормализации
model = LinearRegression().fit(X, y)

# Предсказание
y_pred = model.predict(X)

# Визуализация данных и модели
plt.scatter(X, y, label='Данные')
plt.plot(X, y_pred, color='red', linewidth=2, label='Модель')
plt.xlabel('Признак')
plt.ylabel('Целевая переменная')
plt.legend()
plt.show()
