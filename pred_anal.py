import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from matplotlib.ticker import FuncFormatter

# Загрузка данных
file_path = "Global_YouTube_Statistics.csv"
data = pd.read_csv(file_path)

matplotlib.use('TkAgg')

# Просмотр первых строк данных
print(data.head())

data = data.dropna(subset=['video_views_for_the_last_30_days'])

# Подготовим данные
# Используем подписчиков, категорию и страну для предсказания просмотров за последние 30 дней
X = data[['subscribers', 'category']]  # Признаки
y = data['video views']  # Целевая переменная

# Разделим данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=42)

# Создадим pipeline для обработки категориальных признаков и обучения модели
pipeline = Pipeline([
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['subscribers']),  # Для числового признака 'subscribers' оставляем без изменений
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['category'])  # Преобразуем категориальные признаки с помощью OneHotEncoder
        ]
    )),
    ('regressor', LinearRegression())  # Модель линейной регрессии
])

# Обучим модель
pipeline.fit(X_train, y_train)

# Прогнозируем на тестовой выборке
y_pred = pipeline.predict(X_test)

# Оценим модель
mse = mean_squared_error(y_test, y_pred)  # Среднеквадратичная ошибка
r2 = r2_score(y_test, y_pred)  # Коэффициент детерминации R^2

print(f"Среднеквадратичная ошибка: {mse}")
print(f"Коэффициент детерминации R^2: {r2}")

# Печатаем несколько реальных и предсказанных значений для проверки
comparison = pd.DataFrame({
    'Real Values': y_test,
    'Predicted Values': y_pred
})
print(comparison.head(10))  # Выводим первые несколько значений

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, label='Предсказанные значения')

plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--', label='Идеальные предсказания')

# Подписи для осей и титул
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.title('Сравнение реальных и предсказанных значений')

plt.show()
