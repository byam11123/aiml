# Titanic EDA - Python Script Version
# This script generates all visualizations and insights

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("=" * 70)
print("TITANIC DATASET - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Load the dataset
print("\n[1/10] Loading dataset...")
df = pd.read_csv('Titanic-Dataset.csv')
print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Display first 5 rows
print("\n[2/10] First 5 rows:")
print(df.head())

# Basic info
print("\n[3/10] Dataset Information:")
print(f"  - Shape: {df.shape}")
print(f"  - Columns: {list(df.columns)}")

# Missing values
print("\n[4/10] Missing Values Analysis:")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
for col in df.columns:
    if missing_data[col] > 0:
        print(f"  - {col}: {missing_data[col]} ({missing_percent[col]:.2f}%)")

# Summary statistics
print("\n[5/10] Summary Statistics:")
print(df.describe())

# Visualizations
print("\n[6/10] Generating visualizations...")

# 1. Histograms
df.hist(figsize=(12, 10), bins=20, edgecolor='black', color='skyblue')
plt.suptitle('Distribution of Numerical Features', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('1_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 1_distributions.png")

# 2. Boxplots - using seaborn for better compatibility
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.boxplot(y=df['Age'], ax=axes[0], color='skyblue')
axes[0].set_title('Age Distribution')

sns.boxplot(y=df['Fare'], ax=axes[1], color='lightgreen')
axes[1].set_title('Fare Distribution')

sns.boxplot(y=df['SibSp'], ax=axes[2], color='salmon')
axes[2].set_title('Siblings/Spouses Distribution')
plt.suptitle('Boxplots - Detecting Outliers', fontsize=14)
plt.tight_layout()
plt.savefig('2_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 2_boxplots.png")

# 3. Survival analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(x='Survived', data=df, palette='Set2', ax=axes[0])
axes[0].set_title('Survival Count')
axes[0].set_xticklabels(['Not Survived (0)', 'Survived (1)'])
survival_counts = df['Survived'].value_counts()
axes[1].pie(survival_counts.values, labels=['Not Survived', 'Survived'], 
            autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
axes[1].set_title('Survival Percentage')
plt.suptitle('Target Variable: Survival Analysis', fontsize=14)
plt.tight_layout()
plt.savefig('3_survival_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: 3_survival_analysis.png (Survival Rate: {df['Survived'].mean()*100:.2f}%)")

# 4. Survival by Gender
plt.figure(figsize=(10, 5))
sns.barplot(x='Sex', y='Survived', data=df, palette='Set1')
plt.title('Survival Rate by Gender', fontsize=14)
plt.ylabel('Survival Rate')
plt.xlabel('Gender')
plt.savefig('4_survival_by_gender.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 4_survival_by_gender.png")

# 5. Survival by Class
plt.figure(figsize=(10, 5))
sns.barplot(x='Pclass', y='Survived', data=df, palette='viridis')
plt.title('Survival Rate by Passenger Class', fontsize=14)
plt.ylabel('Survival Rate')
plt.xlabel('Passenger Class')
plt.xticks([0, 1, 2], ['1st Class', '2nd Class', '3rd Class'])
plt.savefig('5_survival_by_class.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 5_survival_by_class.png")

# 6. Survival by Embarked
plt.figure(figsize=(10, 5))
sns.barplot(x='Embarked', y='Survived', data=df, palette='coolwarm')
plt.title('Survival Rate by Embarkation Port', fontsize=14)
plt.ylabel('Survival Rate')
plt.xlabel('Embarkation Port')
plt.savefig('6_survival_by_embarked.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 6_survival_by_embarked.png")

# 7. Age by Survival
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='Survived', kde=True, palette='Set1', bins=30)
plt.title('Age Distribution by Survival', fontsize=14)
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(['Not Survived', 'Survived'])
plt.savefig('7_age_by_survival.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 7_age_by_survival.png")

# 8. Fare by Survival
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Fare', hue='Survived', kde=True, palette='Set1', bins=30)
plt.title('Fare Distribution by Survival', fontsize=14)
plt.xlabel('Fare')
plt.ylabel('Count')
plt.legend(['Not Survived', 'Survived'])
plt.savefig('8_fare_by_survival.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 8_fare_by_survival.png")

# 9. Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn', center=0, 
            fmt='.2f', square=True, linewidths=1)
plt.title('Correlation Matrix', fontsize=16)
plt.savefig('9_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 9_correlation_matrix.png")

# 10. Survival by Class and Gender
plt.figure(figsize=(10, 6))
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df, palette='Set1')
plt.title('Survival Rate by Class and Gender', fontsize=14)
plt.ylabel('Survival Rate')
plt.xlabel('Passenger Class')
plt.xticks([0, 1, 2], ['1st Class', '2nd Class', '3rd Class'])
plt.legend(title='Gender')
plt.savefig('10_survival_by_class_gender.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 10_survival_by_class_gender.png")

# 11. Family Size analysis
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
plt.figure(figsize=(10, 6))
sns.barplot(x='FamilySize', y='Survived', data=df, palette='mako')
plt.title('Survival Rate by Family Size', fontsize=14)
plt.ylabel('Survival Rate')
plt.xlabel('Family Size')
plt.savefig('11_survival_by_family.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 11_survival_by_family.png")

# 12. Age by Class
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Age', data=df, palette='Set2')
plt.title('Age Distribution by Passenger Class', fontsize=14)
plt.ylabel('Age')
plt.xlabel('Passenger Class')
# Convert Pclass numbers to labels
plt.gca().set_xticklabels(['1st Class', '2nd Class', '3rd Class'])
plt.savefig('12_age_by_class.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 12_age_by_class.png")

# Key Insights Summary
print("\n" + "=" * 70)
print("KEY INSIGHTS SUMMARY")
print("=" * 70)

print("\n1. OVERALL SURVIVAL:")
print(f"   - Total passengers: {len(df)}")
print(f"   - Survival rate: {df['Survived'].mean()*100:.2f}%")

print("\n2. GENDER IMPACT:")
survival_by_sex = df.groupby('Sex')['Survived'].mean()
print(f"   - Female survival: {survival_by_sex['female']*100:.2f}%")
print(f"   - Male survival: {survival_by_sex['male']*100:.2f}%")

print("\n3. PASSENGER CLASS IMPACT:")
survival_by_class = df.groupby('Pclass')['Survived'].mean()
print(f"   - 1st Class survival: {survival_by_class[1]*100:.2f}%")
print(f"   - 2nd Class survival: {survival_by_class[2]*100:.2f}%")
print(f"   - 3rd Class survival: {survival_by_class[3]*100:.2f}%")

print("\n4. DATA QUALITY:")
print(f"   - Age missing: {df['Age'].isnull().sum()} ({df['Age'].isnull().sum()/len(df)*100:.2f}%)")
print(f"   - Cabin missing: {df['Cabin'].isnull().sum()} ({df['Cabin'].isnull().sum()/len(df)*100:.2f}%)")
print(f"   - Embarked missing: {df['Embarked'].isnull().sum()}")

print("\n5. CORRELATIONS WITH SURVIVAL:")
corr_with_survival = df.corr(numeric_only=True)['Survived'].sort_values(ascending=False)
for feature in corr_with_survival.index:
    if feature != 'Survived':
        print(f"   - {feature}: {corr_with_survival[feature]:.3f}")

print("\n" + "=" * 70)
print("✓ EDA COMPLETE - All visualizations saved successfully!")
print("=" * 70)
print("\nGenerated Files:")
print("  📊 1_distributions.png - Feature distributions")
print("  📊 2_boxplots.png - Outlier detection")
print("  📊 3_survival_analysis.png - Survival overview")
print("  📊 4_survival_by_gender.png - Gender analysis")
print("  📊 5_survival_by_class.png - Class analysis")
print("  📊 6_survival_by_embarked.png - Embarkation analysis")
print("  📊 7_age_by_survival.png - Age vs survival")
print("  📊 8_fare_by_survival.png - Fare vs survival")
print("  📊 9_correlation_matrix.png - Feature correlations")
print("  📊 10_survival_by_class_gender.png - Combined analysis")
print("  📊 11_survival_by_family.png - Family size analysis")
print("  📊 12_age_by_class.png - Age by class")
print("=" * 70)
