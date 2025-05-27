import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import (homogeneity_score, completeness_score,v_measure_score)
from sklearn.decomposition import PCA




df = pd.read_csv('Global_Music_Streaming_Listener_Preferences.csv')

# 1. Найти страну с максимальным средним возрастом слушателей на всех платформах и жанрах

avg_age_by_country = df.groupby('Country')['Age'].mean()
max_country = avg_age_by_country.idxmax()
max_age = avg_age_by_country.max()
print(f'Страна: {max_country}, максимальный средний возраст: {max_age}.')


# 2. Найти исполнителя, которого наиболее часто случают в обед по подписке Free в жанре Rock

task2 = df[(df['Listening Time (Morning/Afternoon/Night)'] == 'Afternoon') & (df['Subscription Type'] == 'Free') & (df['Top Genre'] == 'Rock')]
artist_counts = task2['Most Played Artist'].value_counts()
print(f'Исполнитель: {artist_counts.idxmax()}')


# 3. Построить круговые диаграммы для каждой платформы с % Free и Premium строками. Сделать вывод, где больше всего Premium

premium_percentages = {}
platforms = df['Streaming Platform'].unique()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, platform in enumerate(platforms):
    subset = df[df['Streaming Platform'] == platform]
    counts = subset['Subscription Type'].value_counts()
    premium_percent = counts.get('Premium') / counts.sum() * 100
    premium_percentages[platform] = premium_percent
    axes[i].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    axes[i].set_title(f'{platform}')

plt.suptitle('Распределение подписок по платформам', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

top_platform = max(premium_percentages, key=premium_percentages.get)
top_value = premium_percentages[top_platform]
print(f"На платформе {top_platform} наибольшая доля Premium-подписчиков: {top_value:.2f}%")


# 4. Найти самого популярного исполнителя и жанр на каждое поколение (по 10 лет от 13 до 60)

results = []
bins = [13, 20, 30, 40, 50, 60, float('inf')]
labels = ['13-19', '20-29', '30-39', '40-49', '50-59', '60+']
df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

for i in labels:
    subset = df[df['Age Group'] == i]
    top_artist = subset['Most Played Artist'].value_counts().idxmax()
    top_genre = subset['Top Genre'].value_counts().idxmax()
    print(f'Возрастная группа: {i}, самый популярный исполнитель: {top_artist}, жанр: {top_genre}.')

# EXTRA - Количественные значения по любимым исполнителям
for i in labels:
    subset = df[df['Age Group'] == i]
    print(f'\nВозрастная группа: {i}\n{subset['Most Played Artist'].value_counts()}')

# 5. Реализовать кластеризацию (по жанрам) на основе признаков: возраст, Listening Time, Repeat Song Rate. Сделать вывод о точности.


df_subset = df[['Age', 'Listening Time (Morning/Afternoon/Night)', 'Repeat Song Rate (%)', 'Top Genre']].copy()
listening_encoder = LabelEncoder()
df_subset['Listening Time (Morning/Afternoon/Night)'] = listening_encoder.fit_transform(df_subset['Listening Time (Morning/Afternoon/Night)'])
n_clusters = df_subset['Top Genre'].nunique()

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(df_subset.drop(columns=['Top Genre']))

true_labels = LabelEncoder().fit_transform(df_subset['Top Genre'])
homogeneity = homogeneity_score(true_labels, cluster_labels)
completeness = completeness_score(true_labels, cluster_labels)
v_measure = v_measure_score(true_labels, cluster_labels)

print(f"Гомогенность: {homogeneity:.3f}")
print(f"Полнота: {completeness:.3f}")
print(f"V-Мера: {v_measure:.3f}")

X = df_subset.drop(columns=['Top Genre'])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7, edgecolor='k')
axes[0].set_title('Кластеры KMeans (PCA)')
axes[0].set_xlabel('PCA 1')
axes[0].set_ylabel('PCA 2')
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=true_labels, cmap='tab10', alpha=0.7, edgecolor='k')
axes[1].set_title('Реальные жанры (PCA)')
axes[1].set_xlabel('PCA 1')
axes[1].set_ylabel('PCA 2')
plt.tight_layout()
plt.show()

# 6 EXTRA

# График, отражающий среднее количество минут прослушивания в день по возрастным группам (из предыдущего задания)

bins = [13, 20, 30, 40, 50, 60, float('inf')]
labels = ['13-19', '20-29', '30-39', '40-49', '50-59', '60+']
df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
grouped = df.groupby('Age Group', observed=True)['Minutes Streamed Per Day'].mean()

plt.figure(figsize=(8, 5))
grouped.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Среднее количество минут прослушивания в день по возрастным группам')
plt.ylabel('Минут в день')
plt.xlabel('Возрастная группа')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# График, отражающий распределение пользователей по странам

country_counts = df['Country'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(country_counts, labels=country_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Распределение пользователей по странам')
plt.axis('equal')
plt.tight_layout()
plt.show()

# График, отражающий среднее значение процента повторного прослушивания песен по возрастным группам
bins = [13, 20, 30, 40, 50, 60, float('inf')]
labels = ['13-19', '20-29', '30-39', '40-49', '50-59', '60+']
df['Age Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
repeat_rate_by_group = df.groupby('Age Group', observed=True)['Repeat Song Rate (%)'].mean()

plt.figure(figsize=(8, 5))
repeat_rate_by_group.plot(kind='bar', color='salmon', edgecolor='black')
plt.title('Среднее значение Repeat Song Rate (%) по возрастным группам')
plt.xlabel('Возрастная группа')
plt.ylabel('Repeat Song Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


