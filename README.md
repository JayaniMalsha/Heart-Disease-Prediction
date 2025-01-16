# Heart-Disease-Prediction
Build a model to predict whether a person has heart disease based on medical features such as age, sex, cholesterol levels, and blood pressure

##Project Overview
This project aims to predict the likelihood of a person having heart disease based on their medical features. Using machine learning techniques, we analyze the dataset and train a model to classify individuals into two categories:

0: No heart disease
1: Heart disease.
The model will be helpful for early detection and intervention, enabling better patient outcomes.

##Languages and Technologies
Languages:
Python: Used for data processing, analysis, and machine learning.
Libraries and Frameworks:
Pandas: For data manipulation and analysis.
NumPy: For numerical computations.
Scikit-learn: For implementing machine learning algorithms, model evaluation, and preprocessing.
Matplotlib & Seaborn: For data visualization and exploratory analysis.
Joblib: For saving and loading the trained models.
Tools and Platforms:
Jupyter Notebook: For interactive development and experimentation.
Kaggle: Source of the dataset and community sharing.
Git: For version control and collaboration.
IDE/Text Editor: Any preferred Python IDE like PyCharm, VSCode, or JupyterLab.
Dataset
Dataset Source: Kaggle Heart Disease Dataset

##Dataset Description
The dataset contains 14 features that are commonly used in medical diagnosis:

age: Age of the individual (years).
sex: Gender of the individual (1 = male, 0 = female).
cp: Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic).
trestbps: Resting blood pressure (mm Hg).
chol: Serum cholesterol level (mg/dl).
fbs: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false).
restecg: Resting electrocardiographic results (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy).
thalach: Maximum heart rate achieved.
exang: Exercise-induced angina (1 = yes, 0 = no).
oldpeak: ST depression induced by exercise relative to rest.
slope: Slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping).
ca: Number of major vessels (0–3) colored by fluoroscopy.
thal: Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect).
target: Target variable (1 = heart disease, 0 = no heart disease).
Project Workflow
Data Preprocessing

Load and explore the dataset.
Handle missing or null values (if any).
Encode categorical variables.
Normalize features using StandardScaler.
Exploratory Data Analysis (EDA)

Analyze feature distributions.
Explore correlations between features and the target variable.
Visualize important patterns using charts and plots.
Model Training

Split the data into training and testing sets (e.g., 80% train, 20% test).
Train multiple machine learning models:
Logistic Regression
Random Forest
Support Vector Machines (SVM)
K-Nearest Neighbors (KNN)
Evaluate model performance using metrics:
Accuracy
Precision, Recall, F1 Score
Confusion Matrix
Hyperparameter Tuning

Use GridSearchCV or RandomizedSearchCV to optimize the Random Forest model.
Model Evaluation

Compare model performance.
Select the best-performing model (Random Forest in this case).
Model Deployment

Save the trained model using joblib.
Test the model on new, unseen data to validate its accuracy.
How to Run the Project
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
git clone https://github.com/your-repo/heart-disease-prediction.git
cd heart-disease-prediction
Download the dataset:

Download the dataset from Kaggle.
Place the dataset file (heart.csv) in the project directory.
Run the training script:

bash
Copy
Edit
python train_model.py
Test the model with new data:

Use the predict.py script to input new data and make predictions:
bash
Copy
Edit
python predict.py
Results
Best Model: Random Forest Classifier.
Accuracy: ~98.5% on the test dataset.
F1 Score: ~0.99.
Cross-Validation Score: ~99.4%.
Example Prediction
For the input:

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
The model predicts:

Probability of Heart Disease: 0.8 (80%)
Prediction: Heart Disease (1)
Files in the Repository
train_model.py: Contains the code for data preprocessing, model training, and evaluation.
predict.py: Contains the code for loading the trained model and making predictions on new data.
heart.csv: Dataset file (to be downloaded from Kaggle).
README.md: Project documentation.
Future Work
Experiment with more advanced models like Gradient Boosting or Neural Networks.
Deploy the model as a REST API using Flask or FastAPI.
Build a web-based interface for user interaction.
Expand the dataset to include more diverse data for better generalization.
License
This project is licensed under the MIT License. See the LICENSE file for more details.
