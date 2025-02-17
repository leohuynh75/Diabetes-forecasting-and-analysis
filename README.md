# Diabetes-forecasting-and-analysis
![](images/images.jpg)
## Introduction
In the recent times, the prevalence of people with diabetes is proliferating, which has posed a formidable challenge to the global healthcare system. To remedy this situation, innovative approaches are necessary to improve the capability of prediction and management this kind of disease. Thus, in this project, I would like to build a Machine Learning model to forecast diabetes, as well as analyze in depth how factocs affect the likelihood of devlopment of disease, so that the doctors can take timely and appropriate measures to prevent and treat the disease effectively.

## Data source
The dataset I use derives from Kaggle (https://www.kaggle.com/datasets/imtkaggleteam/diabetes/data)

## Data overview
The dataset contains the raw data of several hundred rural African-American. This data includes 403 samples and 18 features. Specifically as follows:
  - chol: Total Cholesterol
  - stab.glu: Stabilized Glucose
  - hdl: High Density Lipoprotein Cholesterol
  - ratio: Ratio between Total Cholesterol and HDL (chol/hdl)
  - Age and gender
  - Height and weight
  - Frame: Body Frame Size (Large, Medium, Small)
  - bp.1s: First systolic blood pressure
  - bp.1d: First diastolic blood pressure
  - waist
  - hip: Hybrid Insulin Peptides
  - time.ppn: Partial parenteral nutrition
  - glyhb: Glycosylated Hemoglobin (HBA1C)
## Tools and techniques applied
### 1. Tools
- Google Colabs: using Python to load data, clean data, build Machine Learning models and do feature analysis.
- Power Bi: visualizing the dataset.
### 2. Techniques
In this project, I will use two Machine Learning models: **Random Forest and XGBoost**, to forecast diabetes. Besides, this is a medical dataset, there is an imbalance between classes in target variable, which could lead to classification bias towards the majority class. Thus, I will use **oversampling technique** to solve this problem.
And finally, to do feature analysis, I will choose **SHAPLEY** to see how features affect the outcome.
## Methodology
Here is a pipeline of my work:
|![](images/flowchart.png)|
|:--:|
|**Fig.1. Pipeline of my work**|
### 1. Load data
Firsly, I download the dataset from Kaggle on the above link. Then, I will import it into Google Colabs by the code:
```python
df = pd.read_csv('diabetes.csv')
```
And here is an overview of the dataset:
|![](images/overview.png)|
|:--:|
|**Fig.2. First 5 rows of the dataset**|

### 2. EDA
Now, let's take a look at some descriptive information of this dataset:
```python
df.describe()
```
|![](images/description.png)|
|:--:|
|**Fig.3. Statistical information of numeric features in the dataset**|

From this table, I can draw some information:
  - Missing data appears in many columns, for example: **chol**, **hdl**, **stab.glu**, **glyhb**, etc. A summary table of the number of missing values in all columns is necessary.
  - In terms of the distribution of features:
      - Some features have a large range between min and max, for example, the **stab.glu** (Stabilized Glucose) column has a value range from *48* to *385*, **weight** from *99* to *325* (in lbs), etc. This shows the presence of outliers, more visual images are needed to check.
      - The **chol** (Cholesterol) column has an average of about *207.85*, with a standard deviation of *44.44*, showing that the data has a quite large dispersion.
      - The **glyhb** (HBA1C) column has an average value of *5.59*, but the max is up to *16.11*, which may suggest a group of patients with high blood sugar or diabetes.
        
Here is the number of missing values in all columns:

|![](images/missing_values.png)|
|:--:|
|**Fig.4. Number of missing values in all columns**|

Next, let's see the histogram of the dataset to have a clear view of dataset's distribution:
```python
df.hist(figsize=(12, 10), bins=30)
plt.show()
```
|![](images/histogram.png)|
|:--:|
|**Fig.5. Histogram of all numeric features**|

Then, the boxplots to check the presence of outliers:
```python
numeric_col = df.select_dtypes('number').columns.to_list()
numeric_col.remove('id')
df_melted = df[numeric_col].melt(var_name = 'Feature', value_name = 'Value')

plt.figure(figsize=(12, 6))
sns.boxplot(x="Feature", y="Value", data=df_melted)
plt.xticks(rotation=45)  # Rotate the label name 
plt.yscale('log') # Using log scale to see the boxplots of features with small range better
plt.title("Boxplot của nhiều feature")
plt.show()
```
|![](images/boxplots.png)|
|:--:|
|**Fig.6. Boxplots of all numeric features**|

In summary, I must solve the problem of missing values first, then based on the situation of the dataset after cleaning, I will consider whether to handle the outliers or not. Thus, from doing EDA on the dataset, I can have an overview of the dataset to make a plane for the next step, data cleaning. 
### 3. Data cleaning
From the **Fig.4** above, we can see that there are so many missing values in two columns **bp.2s** and **bp.2d**, both are *262*. Thus, my solution for this is remove them from the dataset:
```python
df.drop(['bp.2s','bp.2d'], axis=1, inplace=True)
```
Next, for the remaining numeric features, I will fill NA with the mean value of each column:
```python
numeric_col_na = ['chol','hdl','ratio','glyhb','height','weight','bp.1s','bp.1d','waist','hip','time.ppn']
def fill_missing_value_numeric(col_name):
  for col in col_name:
    df[col] = df[col].fillna(df[col].mean().round( ))
  return df
fill_missing_value_numeric(numeric_col_na)
```
|![](images/cleaning_result.png)|
|:--:|
|**Fig.7. The result after cleaning**|

Only the feature **frame** left has *12* NA values. Since this is a categorical variable, I will fill the NA with the mode value of the column.
```python
df['frame'] = df['frame'].fillna(df['frame'].mode()[0])
```
After having a data file that no longer has any NA values, I will export this file to Excel format to use for visualization steps on Power Bi:
```python
df.to_excel('cleaned_data.xlsx', index=False)
from google.colab import files # Load this package to download the cleaning file
files.download('cleaned_data.xlsx')
```
Next, I will add a **BMI** column from the 2 features **height** (in inch) and **weight** (in lbs) with the aim of helping the model learn the relationship between features better. To do this, I need to change the units first: 1 kg = 0.4536 lbs and 1 inch = 0.0254 m
|![](images/bmi.jpg)|
|:--:|
|**Fig.8. BMI formula**|

```python
# Add BMI column from weight and height
df['BMI'] = (((df['weight']) * 0.4536) / ((df['height'] * 0.0254)**2)).round(1)
```
Then, to bring the problem to a 2-class classification, I will add an **outcome** column calculated based on the **glyhb** feature with the condition that if **glyhb** is greater than or equal to *6.5*, it will be 1 (Diabetes), otherwise it will be 0 (Normal). This will be the target variable of the dataset.
```python
# Add outcome column from glyhb, 0 for Normal and 1 for Diabetes
df['outcome'] = np.where(df['glyhb'] >= 6.5, 1, 0)
```
|![](images/final_data.png)|
|:--:|
|**Fig.9. Two new columns are added to the dataset**|

### 4. EDA after cleaning
To facilitate the model building later, I will conduct the final EDA test. First, I will check the correlation between the features and the target variable. Because the target variable is binary and the features also have 2 data types: continuous and categorical. So I will divide into 2 groups to check the correlation. Group 1 will use the Point Biserial method to check the correlation between the continuous variable features and the target.
```python
# Check correlation
# Continous numeric feature
continuous_numeric = ['chol','stab.glu','hdl','ratio','glyhb','age','height','weight','bp.1s','bp.1d','waist','hip','time.ppn','BMI']

from scipy.stats import pointbiserialr

for col in continuous_numeric:
    corr, p_value = pointbiserialr(df[col], df['outcome'])
    print(f"Correlation between {col} and Outcome: {corr:.4f}, p-value: {p_value:.4f}")
```
And here is the result:
|![](images/result_biserial.png)|
|:--:|
|**Fig.10. The correlation between continous features and target variable**|

