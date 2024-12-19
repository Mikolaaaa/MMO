import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# Загрузка данных
file_path = "books_of_the_decade.csv"  # Путь к файлу с данными
data = pd.read_csv(file_path)

# Просмотр первых строк данных для проверки структуры
print(data.head())

# Преобразуем данные в числовой формат, если есть строки с числами
data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce')  # Преобразуем Rating в числовой тип, пропущенные значения станут NaN
data['Number of Votes'] = pd.to_numeric(data['Number of Votes'], errors='coerce')  # Преобразуем Number of Votes в числовой тип
data['Score'] = pd.to_numeric(data['Score'], errors='coerce')  # Преобразуем Score в числовой тип

# Проверка на наличие пропущенных значений после преобразования
print(data.isnull().sum())  # Покажет количество NaN в каждом столбце

# Удаляем строки с пропущенными значениями (если они есть)
data = data.dropna(subset=['Rating', 'Number of Votes', 'Score'])

# Выбираем только числовые колонки для корреляции
numerical_data = data[['Rating', 'Number of Votes', 'Score']]

# Вычисляем корреляцию между переменными
correlation_matrix = numerical_data.corr()

# Выводим матрицу корреляции
print("Корреляционная матрица:\n", correlation_matrix)

# Визуализируем корреляционную матрицу с помощью тепловой карты
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Корреляционная матрица')
plt.show()
