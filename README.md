<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Heart Disease Prediction ReadMe</title>
</head>
<body>
    <h1>Heart-Disease-Prediction</h1>

    <h2>Project Overview</h2>
    <p>The Heart Disease Prediction project focuses on building a machine learning model to determine whether an individual is at risk of heart disease. By analyzing key medical features such as age, sex, cholesterol levels, blood pressure, and others, the model aims to classify individuals into two categories:</p>
    <ul>
        <li><strong>0</strong>: No heart disease</li>
        <li><strong>1</strong>: Heart disease</li>
    </ul>
    <p>This model can assist in early detection, paving the way for timely medical interventions and improved patient outcomes.</p>

    <h2>Languages and Technologies</h2>
    <h3>Languages:</h3>
    <ul>
        <li>Python</li>
    </ul>

    <h3>Libraries and Frameworks:</h3>
    <ul>
        <li><strong>Pandas</strong>: Data manipulation and analysis.</li>
        <li><strong>NumPy</strong>: Numerical computations.</li>
        <li><strong>Scikit-learn</strong>: Machine learning algorithms and preprocessing.</li>
        <li><strong>Matplotlib & Seaborn</strong>: Data visualization.</li>
        <li><strong>Joblib</strong>: Model serialization and saving.</li>
    </ul>

    <h3>Tools and Platforms:</h3>
    <ul>
        <li><strong>Jupyter Notebook</strong>: Interactive development.</li>
        <li><strong>Kaggle</strong>: Dataset source and collaboration.</li>
        <li><strong>Git</strong>: Version control.</li>
        <li><strong>Any IDE</strong>: Compatible with Python (e.g., PyCharm, VSCode, or JupyterLab).</li>
    </ul>

    <h2>Dataset</h2>
    <p><strong>Source:</strong> <a href="https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/data" target="_blank">Kaggle Heart Disease Dataset</a></p>

    <h3>Features</h3>
    <p>The dataset includes 14 important medical features:</p>
    <ul>
        <li><strong>age</strong>: Age of the individual (years).</li>
        <li><strong>sex</strong>: Gender (1 = male, 0 = female).</li>
        <li><strong>cp</strong>: Chest pain type (0–3).</li>
        <li><strong>trestbps</strong>: Resting blood pressure (mm Hg).</li>
        <li><strong>chol</strong>: Serum cholesterol level (mg/dl).</li>
        <li><strong>fbs</strong>: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false).</li>
        <li><strong>restecg</strong>: Resting electrocardiographic results (0–2).</li>
        <li><strong>thalach</strong>: Maximum heart rate achieved.</li>
        <li><strong>exang</strong>: Exercise-induced angina (1 = yes, 0 = no).</li>
        <li><strong>oldpeak</strong>: ST depression induced by exercise relative to rest.</li>
        <li><strong>slope</strong>: Slope of the peak exercise ST segment (0–2).</li>
        <li><strong>ca</strong>: Number of major vessels (0–3) colored by fluoroscopy.</li>
        <li><strong>thal</strong>: Thalassemia (1–3).</li>
        <li><strong>target</strong>: Outcome (1 = heart disease, 0 = no heart disease).</li>
    </ul>

    <h2>Workflow</h2>
    <h3>1. Data Preprocessing</h3>
    <ul>
        <li>Loading the dataset.</li>
        <li>Handling missing/null values.</li>
        <li>Encoding categorical features.</li>
        <li>Scaling numerical data with StandardScaler.</li>
    </ul>

    <h3>2. Exploratory Data Analysis (EDA)</h3>
    <ul>
        <li>Visualizing feature distributions and correlations.</li>
        <li>Analyzing the relationship between features and the target variable.</li>
    </ul>

    <h3>3. Model Training</h3>
    <ul>
        <li>Splitting the data into training and testing sets (e.g., 80/20 split).</li>
        <li>Training multiple machine learning algorithms:
            <ul>
                <li>Logistic Regression</li>
                <li>Random Forest</li>
                <li>Support Vector Machines (SVM)</li>
                <li>K-Nearest Neighbors (KNN)</li>
            </ul>
        </li>
        <li>Evaluating models using metrics:
            <ul>
                <li>Accuracy</li>
                <li>Precision, Recall, F1 Score</li>
                <li>Confusion Matrix</li>
            </ul>
        </li>
    </ul>

    <h3>4. Hyperparameter Tuning</h3>
    <ul>
        <li>Optimizing model parameters with GridSearchCV or RandomizedSearchCV.</li>
    </ul>

    <h3>5. Model Evaluation</h3>
    <ul>
        <li>Comparing all models.</li>
        <li>Selecting the best-performing model (e.g., Random Forest).</li>
    </ul>

    <h3>6. Model Deployment</h3>
    <ul>
        <li>Saving the trained model using Joblib.</li>
        <li>Making predictions on new, unseen data.</li>
    </ul>

    <h2>Results</h2>
    <h3>Best Model: Random Forest Classifier</h3>
    <ul>
        <li>Accuracy: ~98.5% on the test dataset</li>
        <li>F1 Score: ~0.99</li>
        <li>Cross-Validation Score: ~99.4%</li>
    </ul>

    <h3>Example Prediction:</h3>
    <pre><code>
Input:
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
    </code></pre>

    <h2>How to Run the Project</h2>
    <h3>Prerequisites</h3>
    <p>Install the required Python libraries:</p>
    <pre><code>
pip install pandas numpy scikit-learn matplotlib seaborn joblib
    </code></pre>

    <h3>Steps to Run</h3>
    <ol>
        <li>Clone the repository:
            <pre><code>git clone https://github.com/JayaniMalsha/Heart-Disease-Prediction.git</code></pre>
            <pre><code>cd heart-disease-prediction</code></pre>
        </li>
        <li>Download the dataset:
            <p>Get the dataset from <a href="https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/data" target="_blank">Kaggle</a>.</p>
            <p>Place heart.csv in the project directory.</p>
        </li>
        <li>Run the training script:
            <pre><code>python train_model.py</code></pre>
        </li>
        <li>Test predictions with new data:
            <pre><code>python predict.py</code></pre>
        </li>
    </ol>

    <h2>Future Work</h2>
    <ul>
        <li>Test advanced algorithms like Gradient Boosting or Neural Networks.</li>
        <li>Deploy the model via Flask or FastAPI.</li>
        <li>Develop a user-friendly web interface.</li>
        <li>Incorporate additional datasets for improved accuracy.</li>
    </ul>

    <h2>License</h2>
    <p>This project is licensed under the MIT License.</p>
</body>
</html>
