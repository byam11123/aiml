# ============================================================
# Titanic Dataset - Data Cleaning & Preparation for ML
# ============================================================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# STEP 1: IMPORT & EXPLORE DATA
# ============================================================

# Load data
df = pd.read_csv('Titanic-Dataset.csv')

# Explore the data
print("=" * 50)
print("STEP 1: DATA EXPLORATION")
print("=" * 50)
print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Dataset Shape (rows, columns) ---")
print(df.shape)

print("\n--- Data Types & Missing Values ---")
print(df.info())

print("\n--- Basic Statistics ---")
print(df.describe())

print("\n--- Missing Values Count ---")
print(df.isnull().sum())

print("\n--- Missing Values Percentage ---")
missing_pct = (df.isnull().sum() / len(df)) * 100
print(missing_pct[missing_pct > 0])

print("\n--- Duplicate Rows ---")
print(f"Duplicates: {df.duplicated().sum()}")

# ============================================================
# STEP 2: HANDLE MISSING VALUES
# ============================================================
print("\n" + "=" * 50)
print("STEP 2: HANDLING MISSING VALUES")
print("=" * 50)

# --- Strategy for Age (19.87% missing) ---
# Use median to fill (robust to outliers)
age_median = df['Age'].median()
print(f"\n--- Filling Age with median: {age_median} ---")
df.fillna({'Age': age_median}, inplace=True)

# --- Strategy for Cabin (77.10% missing) ---
# Too many missing values → Drop the column
print("\n--- Dropping Cabin column (77% missing) ---")
df.drop(columns=['Cabin'], inplace=True)

# --- Strategy for Embarked (0.22% missing) ---
# Only 2 missing → Fill with mode (most frequent value)
embarked_mode = df['Embarked'].mode()[0]
print(f"\n--- Filling Embarked with mode: '{embarked_mode}' ---")
df.fillna({'Embarked': embarked_mode}, inplace=True)

# Verify all missing values are handled
print("\n--- Missing Values After Cleaning ---")
print(df.isnull().sum())

print("\n✅ All missing values handled!")

# ============================================================
# STEP 3: ENCODE CATEGORICAL FEATURES
# ============================================================
print("\n" + "=" * 50)
print("STEP 3: ENCODING CATEGORICAL FEATURES")
print("=" * 50)

# Drop columns not useful for ML
print("\n--- Dropping Name, Ticket, PassengerId ---")
df.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)

# --- Label Encoding for Sex (binary: male=0, female=1) ---
print("\n--- Label Encoding: Sex ---")
print(f"Before: {df['Sex'].value_counts().to_dict()}")
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
print(f"After:  {df['Sex'].value_counts().to_dict()}")

# --- One-Hot Encoding for Embarked (3+ categories) ---
print("\n--- One-Hot Encoding: Embarked ---")
print(f"Before: {df['Embarked'].value_counts().to_dict()}")
df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked', drop_first=False)
# Convert boolean to int (True→1, False→0)
df['Embarked_C'] = df['Embarked_C'].astype(int)
df['Embarked_Q'] = df['Embarked_Q'].astype(int)
df['Embarked_S'] = df['Embarked_S'].astype(int)

print("\n--- DataFrame After Encoding ---")
print(df.head())
print(f"\nColumns: {list(df.columns)}")
print(f"Shape: {df.shape}")

# ============================================================
# STEP 4: NORMALIZE/STANDARDIZE NUMERICAL FEATURES
# ============================================================
print("\n" + "=" * 50)
print("STEP 4: STANDARDIZATION (Z-Score Normalization)")
print("=" * 50)

from sklearn.preprocessing import StandardScaler

# Columns to scale (continuous numerical features)
cols_to_scale = ['Age', 'Fare', 'SibSp', 'Parch']

print("\n--- Before Scaling ---")
print(df[cols_to_scale].describe())

# Initialize scaler
scaler = StandardScaler()

# Fit and transform
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

print("\n--- After Scaling (mean≈0, std≈1) ---")
print(df[cols_to_scale].describe())

print("\n✅ Standardization complete!")

# ============================================================
# STEP 5: DETECT & REMOVE OUTLIERS
# ============================================================
print("\n" + "=" * 50)
print("STEP 5: OUTLIER DETECTION & REMOVAL (IQR Method)")
print("=" * 50)

# --- Visualize outliers using boxplot ---
print("\n--- Creating boxplots for numerical columns ---")
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Boxplots - Outlier Detection (Before Removal)')

for ax, col in zip(axes.flatten(), cols_to_scale):
    ax.boxplot(df[col])
    ax.set_title(col)
    ax.set_ylabel('Standardized Value')

plt.tight_layout()
plt.savefig('outliers_boxplot.png', dpi=150)
print("✅ Boxplot saved as 'outliers_boxplot.png'")
plt.close()

# --- IQR Method to detect outliers ---
print("\n--- Detecting outliers using IQR method ---")
outlier_mask = pd.Series([False] * len(df))

for col in cols_to_scale:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR  # 3x IQR for less aggressive removal
    upper_bound = Q3 + 3 * IQR
    
    col_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
    outlier_mask = outlier_mask | col_outliers
    
    print(f"{col}: {col_outliers.sum()} outliers (bounds: {lower_bound:.2f} to {upper_bound:.2f})")

print(f"\nTotal rows flagged as outliers: {outlier_mask.sum()}")

# Remove outliers (keep only non-outlier rows)
df_clean = df[~outlier_mask].copy()
print(f"Rows before removing outliers: {len(df)}")
print(f"Rows after removing outliers:  {len(df_clean)}")
print(f"Outliers removed: {len(df) - len(df_clean)}")

df = df_clean  # Update main dataframe
print(f"\n✅ Outlier removal complete!")

# ============================================================
# STEP 6: SAVE CLEANED DATA + VISUALIZATIONS
# ============================================================
print("\n" + "=" * 50)
print("STEP 6: SAVING CLEANED DATA & VISUALIZATIONS")
print("=" * 50)

# --- Save cleaned dataset ---
df.to_csv('cleaned_titanic.csv', index=False)
print("\n✅ Cleaned data saved to 'cleaned_titanic.csv'")
print(f"   Final shape: {df.shape}")
print(f"   Final columns: {list(df.columns)}")

# --- Create visualization: Survival by Sex ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Survival distribution
df['Survived'].value_counts().plot(kind='bar', ax=axes[0], color=['#ff6b6b', '#51cf66'])
axes[0].set_title('Survival Distribution')
axes[0].set_xticklabels(['Died', 'Survived'], rotation=0)
axes[0].set_ylabel('Count')

# Plot 2: Age distribution
df['Age'].hist(bins=20, ax=axes[1], color='#4ecdc4', edgecolor='black')
axes[1].set_title('Age Distribution (After Cleaning)')
axes[1].set_xlabel('Age (Standardized)')
axes[1].set_ylabel('Frequency')

# Plot 3: Fare distribution
df['Fare'].hist(bins=20, ax=axes[2], color='#ffe66d', edgecolor='black')
axes[2].set_title('Fare Distribution (After Cleaning)')
axes[2].set_xlabel('Fare (Standardized)')
axes[2].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('cleaned_data_summary.png', dpi=150)
print("✅ Summary visualization saved as 'cleaned_data_summary.png'")
plt.close()

# --- Correlation Heatmap ---
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
            square=True, linewidths=1)
plt.title('Feature Correlation Heatmap (After Cleaning)')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
print("✅ Correlation heatmap saved as 'correlation_heatmap.png'")
plt.close()

print("\n" + "=" * 50)
print("🎉 ALL STEPS COMPLETE!")
print("=" * 50)
print("\n📁 Generated Files:")
print("   1. cleaned_titanic.csv  →  ML-ready dataset")
print("   2. outliers_boxplot.png  →  Outlier visualization")
print("   3. cleaned_data_summary.png  →  Data summary")
print("   4. correlation_heatmap.png  →  Feature correlations")
