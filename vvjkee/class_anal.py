import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib

matplotlib.use('TkAgg')

# Шаг 1: Загрузка данных
file_path = "books_of_the_decade.csv"
data = pd.read_csv(file_path)
data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce')  # Преобразуем Number of Votes в числовой тип
data['Score'] = pd.to_numeric(data['Score'], errors='coerce')  # Преобразуем Score в числовой тип
# Отбираем нужные столбцы для кластеризации
columns_of_interest = ["Rating", "Score"]
filtered_data = data[columns_of_interest]

# Шаг 2: Обработка пропущенных значений
# Удаляем строки с пропущенными значениями
filtered_data = filtered_data.dropna()

# Или можно заполнить пропущенные значения средним значением для каждого столбца
filtered_data = filtered_data.fillna(filtered_data.mean())

# Шаг 3: Стандартизация данных
scaler = StandardScaler()
scaled_data = scaler.fit_transform(filtered_data)

# Шаг 4: Определение оптимального числа кластеров с помощью метода локтя
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Визуализация метода локтя
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Метод локтя для выбора числа кластеров')
plt.xlabel('Количество кластеров')
plt.ylabel('Интертия (внутреннее отклонение)')
plt.show()

# Шаг 5: Применение K-means с выбранным количеством кластеров
optimal_k = 3  # Предположим, что из метода локтя оптимальное количество кластеров - 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

data = data.dropna()  # Заполнить пропущенные значения нулями

# Добавляем информацию о кластерах в исходный DataFrame
data['Cluster'] = clusters

# Шаг 6: Визуализация кластеров с помощью PCA (уменьшаем размерность до 2 для визуализации)
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_data)

# Визуализация кластеров
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=data['Cluster'], palette='Set1', s=100, alpha=0.7)
plt.title('Кластеризация данных с использованием K-means (PCA)')
plt.xlabel('Первая главная компонента')
plt.ylabel('Вторая главная компонента')
plt.legend(title='Кластеры')
plt.show()
