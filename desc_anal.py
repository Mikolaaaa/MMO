import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


# Загрузка данных
file_path = "Global_YouTube_Statistics.csv"
data = pd.read_csv(file_path)

# Удаляем строки с пропущенными значениями
data = data.dropna(subset=['video_views_for_the_last_30_days'])

# Выбор переменных
X = data[['video_views_for_the_last_30_days', 'category', 'subscribers']]
y = data['Country']  # Целевая переменная

# Кодируем категориальную переменную 'category' в числовую
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
categories_encoded = encoder.fit_transform(data[['category']])

# Объединяем числовые и закодированные данные
X_encoded = np.hstack([data[['video_views_for_the_last_30_days', 'subscribers']].values, categories_encoded])

# Масштабируем данные
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Кодируем целевую переменную
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Создаем и обучаем модель дискриминантного анализа
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Предсказываем классы
y_pred = lda.predict(X_test)

# Получаем уникальные классы из тестовой выборки
unique_classes_test = np.unique(y_test)

# Получаем метки для этих классов
target_names = label_encoder.inverse_transform(unique_classes_test)

target_names = [str(name) for name in target_names]

# Генерируем отчет классификации, явно указав метки
print("Классификационный отчет:")
print(classification_report(y_test, y_pred, target_names=target_names, labels=unique_classes_test))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Матрица ошибок')
plt.xlabel('Предсказанные метки')
plt.ylabel('Истинные метки')
plt.tight_layout()
plt.show()

# Визуализация на первых двух дискриминантных компонентах
X_lda = lda.transform(X_scaled)
plt.figure(figsize=(12, 6))
for class_idx in np.unique(y_encoded):
    plt.scatter(X_lda[y_encoded == class_idx, 0], X_lda[y_encoded == class_idx, 1], label=label_encoder.classes_[class_idx], alpha=0.6)
plt.title('Распределение данных на дискриминантных компонентах')
plt.xlabel('Первая дискриминантная компонента')
plt.ylabel('Вторая дискриминантная компонента')
plt.legend()
plt.tight_layout()
plt.show()