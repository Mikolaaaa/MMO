import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import LogLocator

matplotlib.use('TkAgg')

# Шаг 1: Загрузка данных
file_path = "Global_YouTube_Statistics.csv"
data = pd.read_csv(file_path)

# Просмотр первых строк данных
print(data.head())

# Шаг 2: Отбор только нужных столбцов для корреляции
columns_of_interest = ["subscribers", "video views", "video_views_for_the_last_30_days", "uploads"]
filtered_data = data[columns_of_interest]

# Шаг 3: Матрица корреляции для выбранных столбцов
correlation_matrix = filtered_data.corr()

# Шаг 4: Визуализация матрицы корреляции с помощью тепловой карты
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5, fmt=".2f")
plt.title('Матрица корреляции для подписчиков, просмотров и загрузок')
plt.tight_layout()
plt.show()
