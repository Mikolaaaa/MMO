import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import numpy as np
import matplotlib

matplotlib.use('TkAgg')

# Загрузка данных
file_path = "Global_YouTube_Statistics.csv"
data = pd.read_csv(file_path)

# Удаляем строки с пропущенными значениями по целевой переменной
data = data.dropna(subset=['video_views_for_the_last_30_days'])

# Выбираем переменные для анализа
X = data[['subscribers', 'video views', 'category', 'Country']]

# Преобразуем категориальные переменные в числовые
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', ['subscribers', 'video views']),  # Числовые переменные
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['category', 'Country'])  # Категориальные переменные
    ]
)

# Применяем PCA
pca = Pipeline([
    ('preprocessor', preprocessor),
    ('pca', PCA(n_components=3))  # Указываем 3 главные компоненты
])

# Применяем преобразования и PCA
X_pca = pca.fit_transform(X)

# Визуализация объясненной дисперсии
explained_variance = pca.named_steps['pca'].explained_variance_ratio_

# Диаграмма разброса с кластерами
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, c=X_pca[:, 2], cmap='coolwarm')
plt.colorbar(scatter, label='Главная компонента 3')

# Добавляем эллипсы для визуализации кластеров
mean_x = np.mean(X_pca[:, 0])
mean_y = np.mean(X_pca[:, 1])
covariance = np.cov(X_pca[:, 0], X_pca[:, 1])
ellipse = Ellipse((mean_x, mean_y),
                  width=2 * np.sqrt(covariance[0, 0]),
                  height=2 * np.sqrt(covariance[1, 1]),
                  edgecolor='red',
                  facecolor='none',
                  linewidth=2)
plt.gca().add_patch(ellipse)

plt.title('Диаграмма разброса по первым двум главным компонентам')
plt.xlabel('Главная компонента 1')
plt.ylabel('Главная компонента 2')
plt.show()
