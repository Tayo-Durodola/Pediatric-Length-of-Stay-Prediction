### Project Name: Predicting Hospital Length of Stay in Pediatric Appendicitis Using Machine Learning

### Description

This project develops and tests machine learning models to predict whether children with suspected appendicitis will have a short hospital stay (less than 4 days) or a long stay (4 days or more). The code uses a dataset with clinical, laboratory, and ultrasound data to train and evaluate four models: **Linear Regression**, **Random Forest Regressor**, **XGBoost Regressor**, and **LightGBM Regressor**. The goal is to provide a tool that helps hospitals plan care and manage resources effectively by identifying patients likely to require longer hospitalization. The code includes data preprocessing, model training, performance evaluation, and visualization of results.

### Dataset Information

* **Source:** The dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis) and is publicly available via [Zenodo (DOI: 10.5281/zenodo.7669442)](https://doi.org/10.5281/zenodo.7669442). It was collected retrospectively from 782 pediatric patients admitted with abdominal pain at Children’s Hospital St. Hedwig in Regensburg, Germany, between 2016 and 2021.
* **Content:** Includes variables such as age, weight, clinical signs (e.g., fever, tenderness), laboratory results (e.g., white blood cell count, C-reactive protein), and ultrasound findings.
* **File:** The dataset is expected to be an Excel file named `Pediatric Appendicitis.xlsx - All cases.csv`, located in the same directory as the script.
* **Target Variable:** Length of stay (LOS), converted into a binary variable (`Long_Stay_Binary`) based on a median threshold of approximately 4 days.

### Code Information

* **Language:** Python (version 3.x)
* **Files:** A single Python script containing all code for data loading, preprocessing, model development, evaluation, and visualization.
* **Output:** Generates confusion matrix images (`confusion_matrix_[model_name].png`) for each evaluated model and a distribution plot of the length of stay categories (`length_of_stay_distribution.png`). Performance metrics (Accuracy, Precision, Recall, F1-Score) are printed to the console.

### Usage Instructions

* **Setup:** Download or clone the repository and ensure the `Pediatric Appendicitis.xlsx - All cases.csv` dataset is placed in the same directory as the script.
* **Run the Code:** Open the script in a Python environment (e.g., Jupyter Notebook or Google Colab) and execute it. The code will load the data, process it, train models, and display results.
* **View Results:** Check the console for printed performance metrics. View generated image files (PNGs) in the working directory for visualizations.
* **Reproduce Results:** Use the same random seed (`random_state=42`) and dataset to replicate the exact outcomes.

### Requirements

* **Python Libraries:**
    * `pandas` for data manipulation
    * `numpy` for numerical operations
    * `matplotlib` for plotting
    * `seaborn` for enhanced visualizations
    * `scikit-learn` for machine learning tasks
    * `xgboost` (optional, for XGBoost Regressor)
    * `lightgbm` (optional, for LightGBM Regressor)
* **Environment:** Recommended to run in Google Colab or a local Jupyter Notebook environment.
* **Dependencies:** Install required libraries using `pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm` if not already present. Note that `xgboost` and `lightgbm` are optional; the script will gracefully skip their evaluation if they are not installed.

### Methodology

* **Data Processing:** Irrelevant columns (e.g., `Diagnosis`, `US_Performed`, `US_Number`, `Diagnosis_Presumptive`, `Management`, `Severity`) are removed. Missing values are standardized to NaN, and categorical variables are converted to numerical form using one-hot encoding. Missing data is imputed using K-Nearest Neighbors with 5 neighbors. The continuous `Length_of_Stay` target is binarized into 'Not Long Stay' (0) and 'Long Stay' (1) based on the dataset's median LOS.
* **Model Development:** The dataset is split into 80% training and 20% testing sets using stratification to maintain the balance of the binary target variable. Four regression models (Linear Regression, Random Forest Regressor, XGBoost Regressor, LightGBM Regressor) are initialized with specific settings (e.g., `n_estimators=100` for Random Forest, `random_state=42` for reproducibility) and trained on the training data. The regression predictions are then converted to binary classification outcomes using a 0.5 threshold.
* **Evaluation:** Models are tested on the unseen test set, and performance is measured using **Accuracy**, **Precision**, **Recall**, and **F1-score**. A **Confusion Matrix** is generated for each model to visualize true positives, true negatives, false positives, and false negatives.
* **Visualization:** Plots include the distribution of Length of Stay categories and confusion matrices for each model.

### Citations

This dataset and research build on work by Marcinkevičs et al. (2021), available via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis) and [Zenodo](https://doi.org/10.5281/zenodo.7669442), which provided the original pediatric appendicitis data.

### License & Contribution Guidelines

* **License:** This code is released under the MIT License, allowing free use, modification, and distribution, provided the original copyright and license notice are included.
* **Contributions:** Contributions are welcome. Please fork the repository, make changes, and submit a pull request with a clear description of your updates. Ensure compatibility with existing code and document any new features.