# Titanic Dataset - Data Cleaning & Preparation for Machine Learning

## 📋 Project Overview

This project demonstrates the complete data preprocessing pipeline to prepare the **Titanic Dataset** for Machine Learning. It covers all essential steps from raw data exploration to creating an ML-ready dataset.

## 🛠️ Tools & Libraries Used

- **Python** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **scikit-learn** - Feature standardization (StandardScaler)

## 📊 Dataset

- **Source**: [Titanic Dataset](https://www.kaggle.com/c/titanic/data)
- **Original Records**: 891 passengers
- **Features**: 12 columns (PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked)

## 🔄 Data Preprocessing Pipeline

### Step 1: Import & Explore Data

- Loaded the dataset and examined its structure
- Identified missing values and data types
- Checked for duplicate rows
- **Key Finding**: 3 columns had missing values (Age: 19.87%, Cabin: 77.10%, Embarked: 0.22%)

### Step 2: Handle Missing Values

| Column       | Strategy                  | Reason                            |
| ------------ | ------------------------- | --------------------------------- |
| **Age**      | Filled with median (28.0) | 19.87% missing - too much to drop |
| **Cabin**    | Dropped column entirely   | 77.10% missing - unreliable       |
| **Embarked** | Filled with mode ('S')    | Only 2 missing values             |

### Step 3: Encode Categorical Features

- **Label Encoding**: `Sex` → male=0, female=1 (binary variable)
- **One-Hot Encoding**: `Embarked` → Embarked_C, Embarked_Q, Embarked_S (3+ categories)
- **Dropped**: Name, Ticket, PassengerId (not useful for ML prediction)

### Step 4: Standardization (Z-Score Normalization)

Applied `StandardScaler` to numerical features:

- **Age, Fare, SibSp, Parch** → Transformed to mean≈0, std≈1
- **Why?** Ensures all features contribute equally to ML models

### Step 5: Outlier Detection & Removal

- **Method**: IQR (Interquartile Range) with 3× multiplier
- **Visualization**: Boxplots for all numerical columns
- **Result**: Removed 240 outliers, kept 651 clean records

### Step 6: Save & Visualize

- Saved cleaned dataset as `cleaned_titanic.csv`
- Generated visualizations:
  - `outliers_boxplot.png` - Outlier detection
  - `cleaned_data_summary.png` - Data distribution
  - `correlation_heatmap.png` - Feature correlations

## 📁 Files in This Repository

| File                       | Description                                     |
| -------------------------- | ----------------------------------------------- |
| `Clean_and_prepare.py`     | Complete preprocessing code                     |
| `Titanic-Dataset.csv`      | Original raw dataset                            |
| `cleaned_titanic.csv`      | Final ML-ready dataset (651 rows × 10 features) |
| `outliers_boxplot.png`     | Boxplot visualization                           |
| `cleaned_data_summary.png` | Summary histograms                              |
| `correlation_heatmap.png`  | Correlation matrix heatmap                      |

## 🚀 How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the script
python Clean_and_prepare.py
```

## 📤 Final Output

- **Original**: 891 rows × 12 columns (with missing values, categorical text)
- **Cleaned**: 651 rows × 10 columns (no missing values, all numerical, standardized)

## 🎓 Interview Q&A Preparation

### Types of Missing Data?

- **MCAR** (Missing Completely At Random): No pattern (e.g., random data loss)
- **MAR** (Missing At Random): Depends on other observed variables
- **MNAR** (Missing Not At Random): Depends on the missing value itself

### How to Handle Categorical Variables?

- **Label Encoding**: For ordinal/binary variables (e.g., Sex → 0/1)
- **One-Hot Encoding**: For nominal variables with 3+ categories (e.g., Embarked)

### Normalization vs Standardization?

- **Normalization (Min-Max)**: Scales to [0, 1] range. Good when data doesn't follow Gaussian.
- **Standardization (Z-Score)**: Scales to mean=0, std=1. Good for Gaussian-like data and models like SVM, Logistic Regression.

### How to Detect Outliers?

- **IQR Method**: Q1 - 1.5×IQR and Q3 + 1.5×IQR (or 3× for less aggressive)
- **Z-Score**: Values with |z| > 3 are outliers
- **Boxplots**: Visual method to identify extreme points

### Why is Preprocessing Important?

- ML models require numerical input (can't process text)
- Missing values cause errors in training
- Unscaled features bias models toward larger ranges
- Outliers can skew model predictions

### One-Hot vs Label Encoding?

- **Label Encoding**: Assigns integers (0, 1, 2...). Can imply false order. Best for binary/ordinal.
- **One-Hot Encoding**: Creates separate binary columns. No implied order. Best for nominal variables.

### How to Handle Imbalanced Data?

- **SMOTE** (Synthetic Minority Over-sampling)
- **Class weights** in models
- **Undersampling** majority class
- **Oversampling** minority class

### Can Preprocessing Affect Model Accuracy?

- **Yes!** Poor preprocessing → poor model performance
- Proper handling of missing values prevents bias
- Correct encoding preserves information
- Scaling ensures fair feature contribution
- Outlier removal prevents model distortion

---

**Author**: Byamkesh Kaiwartya

**Date**: April 2026
