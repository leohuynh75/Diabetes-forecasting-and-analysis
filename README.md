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
|![](images/pipeline.png)|
|:--:|
|**Fig1. Pipeline of my work**|
### 1. Load data
Firsly, I download the dataset from Kaggle on the above link. Then, I will import it into Google Colabs by the code:
```python
df = pd.read_csv('diabetes.csv')
```
And here is an overview of the dataset:

  
