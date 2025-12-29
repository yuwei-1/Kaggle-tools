# Kaggle-tools
A repository for all things machine learning for use in Kaggle competitions

### 🏆 Kaggle Competition Highlights

**[Regression with an Insurance Dataset](https://www.kaggle.com/competitions/playground-series-s4e12)** (Playground S4E12)
* **Rank:** 🥈 **2nd Place** / 2390 Teams
* **Competition:** A regression challenge to predict insurance premiums using a synthetic dataset derived from deep learning models, evaluated on Root Mean Squared Logarithmic Error (RMSLE).
* **My Solution:** I employed a OOF ensembling strategy, generating 118 out-of-fold predictions using AutoML frameworks (AutoGluon, FLAML) and custom tree-based models. These were then combined using a PyTorch TabNet model to create a non-linear meta-ensemble for the final prediction.

**[Loan Approval Prediction](https://www.kaggle.com/competitions/playground-series-s4e10)** (Playground S4E10)
* **Rank:** 🥈 **20th Place** / 3858 Teams (Top 0.5%)
* **Competition:** A binary classification task focused on predicting loan approval status based on applicant financial data, evaluated using the Area Under the ROC Curve (AUC-ROC).

**[CIBMTR - Equity in post-HCT Survival Predictions](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions)**
* **Rank:** 🥈 **150th Place** / 3325 Teams (Silver Medal)
* **Competition:** A research code competition to predict transplant survival rates for HCT patients, aiming to improve equity in healthcare predictions using the Stratified Concordance Index.


python ./ktools/scripts/create_plots_for_binary_classication_problem.py ./data/diabetes_prediction/train.csv diagnosed_diabetes --mapping '{0 : "No diabetes", 1 : "Has diabetes"}' --output_dir ./data/diabetes_prediction/plots/
