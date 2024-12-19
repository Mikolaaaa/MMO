import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import matplotlib

matplotlib.use('TkAgg')

# Загрузка данных
file_path = "Global_YouTube_Statistics.csv"
data = pd.read_csv(file_path)

# Удаляем строки с пропущенными значениями по целевой переменной
data = data.dropna(subset=['video_views_for_the_last_30_days'])

# Подготовим данные
X = data[['subscribers', 'category']]  # Признаки
y = data['video views']  # Целевая переменная

# Разделим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Создаем pipeline для обработки категориальных признаков и обучения модели
pipeline_linear = Pipeline([
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['subscribers']),  # Для числового признака 'subscribers' оставляем без изменений
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['category'])  # Преобразуем категориальные признаки с помощью OneHotEncoder
        ]
    )),
    ('regressor', LinearRegression())  # Модель линейной регрессии
])

# Обучаем модель
pipeline_linear.fit(X_train, y_train)

# Прогнозируем на тестовой выборке
y_pred_linear = pipeline_linear.predict(X_test)

# Оценим модель
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print(f"Линейная регрессия - Среднеквадратичная ошибка: {mse_linear}")
print(f"Линейная регрессия - Коэффициент детерминации R^2: {r2_linear}")


# Создаем pipeline для полиномиальной регрессии
pipeline_poly = Pipeline([
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['subscribers']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['category'])
        ]
    )),
    ('poly', PolynomialFeatures(degree=2)),  # Полиномиальные признаки
    ('regressor', LinearRegression())  # Модель линейной регрессии
])

# Обучаем модель
pipeline_poly.fit(X_train, y_train)

# Прогнозируем на тестовой выборке
y_pred_poly = pipeline_poly.predict(X_test)

# Оценим модель
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"Полиномиальная регрессия - Среднеквадратичная ошибка: {mse_poly}")
print(f"Полиномиальная регрессия - Коэффициент детерминации R^2: {r2_poly}")


# Создаем pipeline для случайного леса
pipeline_rf = Pipeline([
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['subscribers']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['category'])
        ]
    )),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))  # Случайный лес
])

# Обучаем модель
pipeline_rf.fit(X_train, y_train)

# Прогнозируем на тестовой выборке
y_pred_rf = pipeline_rf.predict(X_test)

# Оценим модель
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Случайный лес - Среднеквадратичная ошибка: {mse_rf}")
print(f"Случайный лес - Коэффициент детерминации R^2: {r2_rf}")


# Визуализация результатов линейной регрессии
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_linear, label='Предсказания (Линейная регрессия)', alpha=0.7)
plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--', label='Идеальные предсказания')
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.title('Сравнение реальных и предсказанных значений (Линейная регрессия)')
plt.legend()
plt.show()

# Визуализация результатов полиномиальной регрессии
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_poly, label='Предсказания (Полиномиальная регрессия)', alpha=0.7)
plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--', label='Идеальные предсказания')
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.title('Сравнение реальных и предсказанных значений (Полиномиальная регрессия)')
plt.legend()
plt.show()

# Визуализация результатов случайного леса
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, label='Предсказания (Случайный лес)', alpha=0.7)
plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--', label='Идеальные предсказания')
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.title('Сравнение реальных и предсказанных значений (Случайный лес)')
plt.legend()
plt.show()
