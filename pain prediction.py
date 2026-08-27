import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
df = pd.read_csv("Period_Log.csv")
print(f"Data loaded! {len(df)} rows")

# Analysis
print("\nPain Level Statistics:")
print(f"Average: {df['pain_level'].mean():.2f}")
print(f"Minimum: {df['pain_level'].min()}")
print(f"Maximum: {df['pain_level'].max()}")

pms_pain = df.groupby('pms_symptoms')['pain_level'].mean()
print(f"\nWith PMS: {pms_pain.get('Yes', 'N/A'):.2f}")
print(f"Without PMS: {pms_pain.get('No', 'N/A'):.2f}")

flow_pain = df.groupby('flow_level')['pain_level'].mean()
print(f"\nLight flow pain: {flow_pain.get('Light', 'N/A'):.2f}")
print(f"Moderate flow pain: {flow_pain.get('Moderate', 'N/A'):.2f}")
print(f"Heavy flow pain: {flow_pain.get('Heavy', 'N/A'):.2f}")

# Prepare features
df_simple = df.copy()
df_simple['pms_num'] = (df_simple['pms_symptoms'] == 'Yes').astype(int)
df_simple['flow_num'] = df_simple['flow_level'].map({'Light': 1, 'Moderate': 2, 'Heavy': 3})

X = df_simple[['pms_num', 'flow_num', 'mood_score', 'stress_score_cycle']].dropna()
y = df_simple['pain_level'][X.index]

# Training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"\nMAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

# Saving the charts
sns.set_style("whitegrid")

plt.figure(figsize=(10, 6))
sns.histplot(df['pain_level'], bins=10, kde=True, color='purple')
plt.axvline(df['pain_level'].mean(), color='red', linestyle='--')
plt.title('Distribution of Menstrual Pain Levels')
plt.savefig('pain_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
sns.boxplot(x='pms_symptoms', y='pain_level', data=df)
plt.title('Pain Distribution: PMS vs No PMS')
plt.savefig('pms_vs_pain.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
sns.boxplot(x='cycle_phase', y='pain_level', data=df)
plt.title('Pain Distribution by Cycle Phase')
plt.savefig('cycle_phase_pain.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(x='flow_level', y='pain_level', data=df, order=['Light', 'Moderate', 'Heavy'])
plt.title('Pain Distribution by Flow Level')
plt.savefig('flow_level_pain.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n All charts saved!")

# Interactive predictor
print("\n" + "*" * 60)
print("MENSTRUAL PAIN PREDICTOR")
print("*" * 60)

pms_input = input("Do you have PMS symptoms? (yes/no): ").strip().lower()
flow_input = input("What is your flow level? (light/moderate/heavy): ").strip().lower()
mood_input = float(input("Mood today? (1-10): "))
stress_input = float(input("Stress level? (1-10): "))

pms_num = 1 if pms_input == 'yes' else 0
flow_num = {'light': 1, 'moderate': 2, 'heavy': 3}.get(flow_input, 2)

your_data_df = pd.DataFrame(
    [[pms_num, flow_num, mood_input, stress_input]],
    columns=['pms_num', 'flow_num', 'mood_score', 'stress_score_cycle']
)
predicted_pain = model.predict(your_data_df)[0]

print(f"\nAI PREDICTS YOUR PAIN: {predicted_pain:.1f} / 10")

if predicted_pain <= 3:
    print("Mild pain expected.")
elif predicted_pain <= 6:
    print("Moderate pain expected. Consider rest and pain relief.")
else:
    print("Severe pain expected. Rest and consult a healthcare provider.")