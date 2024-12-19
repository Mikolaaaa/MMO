import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
import matplotlib

matplotlib.use('TkAgg')

# Загрузка данных
file_path = "Global_YouTube_Statistics.csv"
data = pd.read_csv(file_path)

# Удаляем строки с пропущенными значениями
data = data.dropna(subset=['video views', 'category'])

# Группируем данные по категориям
grouped_data = data.groupby('category')['video views']

# Подготовка данных для ANOVA
groups = [group for _, group in grouped_data]

# Проведение однофакторного дисперсионного анализа
anova_result = f_oneway(*groups)
print(f"F-статистика: {anova_result.statistic:.2f}, p-значение: {anova_result.pvalue:.4f}")

# Вывод результатов
if anova_result.pvalue < 0.05:
    print("Различия между категориями статистически значимы (p < 0.05).")
else:
    print("Нет статистически значимых различий между категориями (p >= 0.05).")

# Визуализация: ящик с усами для распределения видео по категориям
plt.figure(figsize=(12, 8))
sns.boxplot(x='category', y='video views', data=data)
plt.title('Распределение просмотров видео по категориям контента')
plt.xticks(rotation=45)
plt.xlabel('Категория контента')
plt.ylabel('Количество просмотров видео')
plt.tight_layout()
plt.show()


