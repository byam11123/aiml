# Task 2: Exploratory Data Analysis (EDA) - Titanic Dataset

## 📋 Overview
This project performs Exploratory Data Analysis (EDA) on the Titanic dataset to understand passenger survival patterns, identify key features, and uncover insights before building machine learning models.

## 📊 Dataset
- **Source**: Titanic Dataset (Titanic-Dataset.csv)
- **Records**: 891 passengers
- **Features**: 12 columns (PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked)

## 🔍 Key Findings

### 1. Overall Survival Rate
- Approximately **38.38%** of passengers survived
- Clear class imbalance in the target variable

### 2. Gender Impact
- **Female passengers** had significantly higher survival rates (~74%)
- **Male passengers** had much lower survival rates (~18%)
- This reflects the "women and children first" protocol

### 3. Passenger Class Impact
- **1st Class**: Highest survival rate (~63%)
- **2nd Class**: Moderate survival rate (~47%)
- **3rd Class**: Lowest survival rate (~24%)
- Strong correlation between socio-economic status and survival

### 4. Age Distribution
- Children (young ages) had higher survival rates
- Age feature has missing values (~20% missing)
- Most passengers were between 20-40 years old

### 5. Fare Distribution
- Fare is **right-skewed** (most passengers paid lower fares)
- Higher fare payers had better survival rates
- Strong positive correlation with survival

### 6. Family Size
- Passengers with moderate family sizes (2-4 members) had better survival rates
- Solo travelers and very large families had lower survival rates

### 7. Embarkation Port
- Passengers who embarked from **Cherbourg (C)** had highest survival rates
- Followed by **Queenstown (Q)**, then **Southampton (S)**

## 📈 Visualizations Generated
1. **Distribution plots** - Histograms of all numerical features
2. **Boxplots** - Outlier detection for Age, Fare, and SibSp
3. **Survival analysis** - Count plots and pie charts
4. **Feature relationships** - Survival rates by:
   - Gender
   - Passenger Class
   - Embarkation Port
   - Age distribution
   - Fare distribution
   - Family Size
   - Class + Gender combination
5. **Correlation heatmap** - Relationships between all numerical features

## 🛠️ Tools Used
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical data visualization
- **NumPy**: Numerical operations

## 📁 Files
- `titanic_eda.ipynb` - Complete EDA Jupyter Notebook with code and visualizations
- `Titanic-Dataset.csv` - Original dataset
- `README.md` - This file (summary of findings)
- Generated visualization images (PNG files)

## 🚀 How to Run
1. Install required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn jupyter
   ```

2. Open the notebook:
   ```bash
   jupyter notebook titanic_eda.ipynb
   ```

3. Run all cells to generate visualizations and insights

## 💡 Insights for ML Modeling
Based on EDA, the following features are likely important for prediction:
1. **Sex** - Strongest predictor of survival
2. **Pclass** - Socio-economic status strongly correlates with survival
3. **Fare** - Higher fares associated with better survival rates
4. **Age** - Children had priority in rescue
5. **FamilySize** - Moderate family sizes show better survival

## ⚠️ Data Quality Issues
- **Cabin**: ~77% missing values (may need to be dropped or engineered)
- **Age**: ~20% missing values (imputation needed)
- **Embarked**: 2 missing values (minimal impact)

## 📝 Conclusion
The EDA reveals clear patterns in the Titanic dataset:
- **Gender and Class** are the strongest predictors of survival
- The data supports historical accounts of "women and children first"
- Socio-economic status (reflected in Pclass and Fare) played a crucial role
- The dataset is suitable for building classification models after handling missing values

---

**Task Completed**: Exploratory Data Analysis ✅  
**Date**: April 10, 2026
