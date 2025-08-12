import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\NajdA\Downloads\archive (4)\HeartDiseaseTrain-Test.csv")
print(df.head())
print(df.isnull().sum())
print(df.info())
print(df.describe())
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
label_cols = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg', 'exercise_induced_angina', 'slope', 'vessels_colored_by_flourosopy', 'thalassemia']

for col in label_cols:
    df[col] = le.fit_transform(df[col])
X = df.drop('target', axis=1)
y = df['target']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(df.head())
print(df[['sex', 'chest_pain_type']].head())
df.to_csv("heart_data_cleaned.csv", index=False)

df.to_csv("heart_data_cleaned.csv", index=False)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# قراءة البيانات
df = pd.read_csv("heart_data_cleaned.csv")

# إنشاء نافذة للرسومات (2 صفوف × 3 أعمدة)
plt.figure(figsize=(18, 10))

# 1️⃣ Gender vs Heart Disease
plt.subplot(2, 3, 1)
sns.countplot(data=df, x='sex', hue='target')
plt.title('Gender vs Heart Disease')

# 2️⃣ Age Distribution
plt.subplot(2, 3, 2)
sns.histplot(data=df, x='age', bins=20, kde=True)
plt.title('Age Distribution')

# 3️⃣ Chest Pain Type vs Disease
plt.subplot(2, 3, 3)
sns.countplot(data=df, x='chest_pain_type', hue='target')
plt.title('Chest Pain Type vs Disease')

# 4️⃣ Age vs Heart Disease (Boxplot)
plt.subplot(2, 3, 4)
sns.boxplot(data=df, x='target', y='age')
plt.title('Age vs Heart Disease')

# 5️⃣ Vessels Count vs Disease
plt.subplot(2, 3, 5)
sns.countplot(data=df, x='vessels_colored_by_flourosopy', hue='target')
plt.title('Vessels Count vs Disease')

plt.tight_layout()
plt.show()

# 6️⃣ Correlation Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


print("✅ Starting evaluation...")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# اقرأ الملف اللي رمّزته (لازم يكون فعلاً مرمّز)
df = pd.read_csv("heart_data_cleaned.csv")

# تأكد إن كل الأعمدة رقمية
print(df.dtypes)

# تقسيم البيانات
X = df.drop("target", axis=1)
y = df["target"]

# فصل البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# بناء وتدريب الموديل
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# التنبؤ
y_pred = model.predict(X_test)

# تقييم النموذج
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.2f}\n")
print("🔍 Classification Report:\n")
print(classification_report(y_test, y_pred))

# مصفوفة الالتباس
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

import pandas as pd

# البيانات اللي استخدمتها في الاختبار
X_test = X_test.copy()  # لو حابب تحفظها بدون مشاكل
X_test['predicted_target'] = y_pred  # أضف عمود التنبؤ

# لو عندك y_test مع X_test وتبي تحفظهم مع بعض
X_test['actual_target'] = y_test.values

# حفظ الملف
X_test.to_csv('heart_data_with_predictions.csv', index=False)
