# Heart-Disease-Prediction
## Project Overview
The Heart Disease Prediction project focuses on building a machine learning model to determine whether an individual is at risk of heart disease. By analyzing key medical features such as age, sex, cholesterol levels, blood pressure, and others, the model aims to classify individuals into two categories:

0: No heart disease
1: Heart disease
This model can assist in early detection, paving the way for timely medical interventions and improved patient outcomes.

## Languages and Technologies
Languages:
Python
Libraries and Frameworks:
Pandas: Data manipulation and analysis.
NumPy: Numerical computations.
Scikit-learn: Machine learning algorithms and preprocessing.
Matplotlib & Seaborn: Data visualization.
Joblib: Model serialization and saving.
Tools and Platforms:
Jupyter Notebook: Interactive development.
Kaggle: Dataset source and collaboration.
Git: Version control.
Any IDE: Compatible with Python (e.g., PyCharm, VSCode, or JupyterLab).
## Dataset
Source: Kaggle Heart Disease Dataset

## Features
The dataset includes 14 important medical features:

age: Age of the individual (years).
sex: Gender (1 = male, 0 = female).
cp: Chest pain type (0–3).
trestbps: Resting blood pressure (mm Hg).
chol: Serum cholesterol level (mg/dl).
fbs: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false).
restecg: Resting electrocardiographic results (0–2).
thalach: Maximum heart rate achieved.
exang: Exercise-induced angina (1 = yes, 0 = no).
oldpeak: ST depression induced by exercise relative to rest.
slope: Slope of the peak exercise ST segment (0–2).
ca: Number of major vessels (0–3) colored by fluoroscopy.
thal: Thalassemia (1–3).
target: Outcome (1 = heart disease, 0 = no heart disease).
## Workflow
### 1. Data Preprocessing
Loading the dataset.
Handling missing/null values.
Encoding categorical features.
Scaling numerical data with StandardScaler.
### 2. Exploratory Data Analysis (EDA)
Visualizing feature distributions and correlations.
Analyzing the relationship between features and the target variable.
### 3. Model Training
Splitting the data into training and testing sets (e.g., 80/20 split).
Training multiple machine learning algorithms:
Logistic Regression
Random Forest
Support Vector Machines (SVM)
K-Nearest Neighbors (KNN)
Evaluating models using metrics:
Accuracy
Precision, Recall, F1 Score
Confusion Matrix
### 4. Hyperparameter Tuning
Optimizing model parameters with GridSearchCV or RandomizedSearchCV.
### 5. Model Evaluation
Comparing all models.
Selecting the best-performing model (e.g., Random Forest).
### 6. Model Deployment
Saving the trained model using Joblib.
Making predictions on new, unseen data.
## Results
Best Model: Random Forest Classifier
Accuracy: ~98.5% on the test dataset
F1 Score: ~0.99
Cross-Validation Score: ~99.4%
Example Prediction:
Input:

python
Copy
Edit
{
    'age': 55,
    'sex': 1,
    'cp': 3,
    'trestbps': 150,
    'chol': 300,
    'fbs': 1,
    'restecg': 0,
    'thalach': 120,
    'exang': 1,
    'oldpeak': 2.5,
    'slope': 2,
    'ca': 1,
    'thal': 3
}
Output:

Probability of Heart Disease: 0.8 (80%)
Prediction: Heart Disease
## How to Run the Project
Prerequisites
Install the required Python libraries:

bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib seaborn joblib
Steps to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/JayaniMalsha/Heart-Disease-Prediction.git
cd heart-disease-prediction
Download the dataset:

Get the dataset from Kaggle.
Place heart.csv in the project directory.
Run the training script:

bash
Copy
Edit
python train_model.py
Test predictions with new data:

bash
Copy
Edit
python predict.py
## Future Work
Test advanced algorithms like Gradient Boosting or Neural Networks.
Deploy the model via Flask or FastAPI.
Develop a user-friendly web interface.
Incorporate additional datasets for improved accuracy.
## License
This project is licensed under the MIT License.
