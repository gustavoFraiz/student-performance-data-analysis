from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
student_performance = fetch_ucirepo(id=320)
df = pd.DataFrame.from_dict(student_performance['data']['original'])

# Basic dataset overview
instances, features = df.shape
print('Instâncias:', instances)
print('Features:', features)

# Univariate exploration
for column in ['G3', 'absences']:
    var = df[column]
    sns.histplot(var, bins=10, kde=True)
    plt.title(column)
    plt.show()
    print('Skewness =', var.skew())
    print('Kurtosis =', var.kurt())
    print('Média =', var.mean())
    print('Desvio padrão =', var.std())

# Final grade vs absences
bins = [0, 5, 10, 20, 30]
labels = ['0-5 faltas', '6-10 faltas', '11-20 faltas', '21-30']
df['grupo_faltas'] = pd.cut(df['absences'], bins=bins, labels=labels)
medias = df.groupby('grupo_faltas')['G3'].mean().reset_index()
contagens = df['grupo_faltas'].value_counts().reset_index()
contagens.columns = ['grupo_faltas', 'n_alunos']
medias = medias.merge(contagens, on='grupo_faltas')

plt.figure(figsize=(9, 5))
bars = sns.barplot(x='grupo_faltas', y='G3', data=medias, palette='Blues_r', edgecolor='black', alpha=0.8)
for bar, n in zip(bars.patches, medias['n_alunos']):
    height = bar.get_height()
    bars.text(bar.get_x() + bar.get_width() / 2, height / 2, f'Média: {height:.1f}\n(n={n})', ha='center', va='center', color='black', fontsize=9)
plt.suptitle('Mais Faltas Correlacionam-se com Notas Mais Baixas?', y=0.98, fontsize=12)
plt.title('Média da nota final (G3) por grupo de faltas | Escala: 0-20', fontsize=9, pad=10)
plt.xlabel('Grupo de Faltas (dias/ano)')
plt.ylabel('Média da Nota Final (G3)')
plt.grid(axis='y', linestyle=':', alpha=0.4)
sns.despine(left=True)
plt.show()

# Family support vs gender
ctab = pd.crosstab(df['sex'], df['famsup'], normalize='index')
sns.heatmap(ctab, annot=True, fmt='.0%', cmap='Blues', linecolor='black')
plt.title('Family Support by Gender')
plt.show()

# Study time vs final grade
pivot_table = df.groupby('studytime')['G3'].mean().reset_index()
study_labels = {1: '<2h', 2: '2-5h', 3: '5-10h', 4: '>10h'}
pivot_table['studytime_label'] = pivot_table['studytime'].map(study_labels)
plt.figure(figsize=(8, 4))
ax = sns.barplot(data=pivot_table, x='studytime_label', y='G3', palette='Blues')
plt.title('Students Who Study More Achieve Higher Final Grades', pad=20)
plt.xlabel('Weekly Study Time')
plt.ylabel('Average Grade (G3)')
plt.ylim(0, 20)
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.show()

# Alcohol consumption vs absences
plt.figure(figsize=(10, 5))
order = [1, 2, 3, 4, 5]
palette = ['#FFEEAD', '#FFCC5C', '#FF8C42', '#E34A33', '#B30000']
ax = sns.barplot(x='Dalc', y='absences', data=df, estimator='mean', errorbar=None, order=order, palette=palette, edgecolor='black', alpha=0.8)
plt.title('Consumo de Álcool em Dias Úteis vs. Faltas', pad=20)
plt.xlabel('Nível de Consumo de Álcool (Dalc)\n1 = Baixo, 5 = Alto')
plt.ylabel('Média de Faltas (dias/ano)')
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height() + 0.2), ha='center', va='center')
plt.grid(axis='y', linestyle=':', alpha=0.3)
sns.despine()
plt.show()

# Binary classification task: approved vs failed
approval_threshold = 10
df['approved'] = (df['G3'] >= approval_threshold).astype(int)
X = df.drop(['G3', 'approved', 'grupo_faltas'], axis=1)
y = df['approved']

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
])

models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, classifier in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [], []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
    print(f'\n{name}')
    print(f'  Mean Accuracy: {np.mean(accuracy_scores):.2f}')
    print(f'  Mean Precision: {np.mean(precision_scores):.2f}')
    print(f'  Mean Recall: {np.mean(recall_scores):.2f}')
    print(f'  Mean F1: {np.mean(f1_scores):.2f}')
