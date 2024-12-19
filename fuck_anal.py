import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


# Загрузка данных
file_path = "Global_YouTube_Statistics.csv"
data = pd.read_csv(file_path)

matplotlib.use('TkAgg')

# Удаляем строки с пропущенными значениями по целевой переменной
data = data.dropna(subset=['video_views_for_the_last_30_days'])

# Выбираем переменные для факторного анализа
X = data[['subscribers', 'category', 'Country', 'video_views_for_the_last_30_days', 'video views']]

# Преобразуем категориальные переменные в числовые
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['subscribers', 'video_views_for_the_last_30_days', 'video views']),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['category', 'Country'])  # Плотный вывод
    ]
)

# Применяем факторный анализ
factor_analysis = Pipeline([
    ('preprocessor', preprocessor),
    ('factor', FactorAnalysis(n_components=3))  # Скажем, что хотим выделить 3 скрытых фактора
])

# Применяем преобразования и факторный анализ
X_transformed = factor_analysis.fit_transform(X)

# Показываем компоненты факторов
print("Компоненты факторов:\n", factor_analysis.named_steps['factor'].components_)

print("Преобразованные данные:\n", X_transformed)

# Визуализация компонентов факторов
components = factor_analysis.named_steps['factor'].components_


X_transformed = factor_analysis.fit_transform(X)

# Получаем имена столбцов после one-hot кодирования
columns_after_transform = preprocessor.transformers_[1][1].get_feature_names_out(['category', 'Country'])

# Объединяем числовые и категориальные столбцы
all_columns = ['subscribers', 'video_views_for_the_last_30_days', 'video views'] + list(columns_after_transform)

# Создаем DataFrame для удобства
components_df = pd.DataFrame(components, columns=all_columns)

# Визуализируем компоненты факторов
plt.figure(figsize=(10, 6))
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=X_transformed[:, 2], cmap='viridis', alpha=0.7)
plt.colorbar(label='Скрытый фактор 3')  # Добавляем цветовую шкалу для третьего скрытого фактора
plt.title('Распределение по первым двум скрытым факторам (с цветом для третьего)')
plt.xlabel('Скрытый фактор 1')
plt.ylabel('Скрытый фактор 2')
plt.show()