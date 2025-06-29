# Breast Cancer Diagnosis Prediction

This repository contains an end‑to‑end data science project that tackles the
**Breast Cancer Wisconsin Diagnostic** problem.  Using features extracted from
digitised images of fine‑needle aspirate (FNA) of breast masses, the goal is
to predict whether a tumour is **malignant** or **benign**.  The problem is
formulated as a binary classification task.  The project demonstrates a
complete workflow including data exploration, preprocessing, model
development, cross‑validation, evaluation and visualisation using
Python’s scientific computing stack.

## Dataset

The dataset used here comes from the `sklearn.datasets` module.  It
originates from the Breast Cancer Wisconsin Diagnostic Dataset.  Each
sample describes the characteristics of cell nuclei present in an FNA
image; 30 real‑valued features such as radius, texture and concavity
are provided.  The target variable indicates whether the tumour is
malignant (encoded as `1`) or benign (`0`).  This dataset is widely
used in machine‑learning research to benchmark classification algorithms
because the features are well‑behaved and the class imbalance is
manageable.  According to published tutorials on the topic, the
dataset is commonly employed to classify tumours based on FNA features
and logistic regression is often used as a baseline classifier【880477803465464†L84-L92】.

For reproducibility, the script included in this repository writes a
copy of the dataset to `breast_cancer_data.csv` in the project root.

## Methodology

### Exploratory Data Analysis

Exploratory data analysis (EDA) was performed using pandas, seaborn
and matplotlib.  Summary statistics give an overview of the scale and
distribution of each feature.  A **correlation heatmap** shows the pairwise
relationships between all 30 features, revealing groups of highly
correlated variables.  To understand how the features relate to the
target, the absolute correlation between each feature and the class
label was computed.  The six most predictive features were visualised
using violin plots to compare their distributions across benign and
malignant cases.  These plots are saved in the `plots/` directory.

### Modelling Approach

The problem is treated as binary classification.  Four classical
supervised algorithms were tested:

1. **Logistic Regression** – a linear model that predicts the
   probability of belonging to a class using the sigmoid function.
   Logistic regression is commonly used for binary classification tasks
   because it outputs probabilities between 0 and 1【625540818750763†L81-L87】.
   It assumes a linear relationship between the predictors and the log‑odds
   of the event occurring and is straightforward to interpret.
2. **Random Forest** – an ensemble of decision trees.  A random
   forest combines the output of many decision trees built on random
   subsets of the data and features and averages their predictions.
   This ensemble method is popular because it reduces overfitting and
   handles both classification and regression problems well【116672908942517†L329-L367】.
3. **Support Vector Machine (RBF kernel)** – a non‑linear classifier
   that constructs a decision boundary by maximising the margin between
   classes.  Using the radial basis function kernel enables the
   algorithm to capture non‑linear relationships.
4. **k‑Nearest Neighbours (k‑NN)** – a simple classifier that assigns
   a class based on the majority class among the `k` closest training
   samples in the feature space.

Each classifier was embedded in a scikit‑learn `Pipeline` with
standard feature scaling (`StandardScaler`) so that all features are
on a comparable scale.  To get the most out of the models, we
performed **hyperparameter tuning** using grid search cross‑validation.
In scikit‑learn, hyper‑parameters (e.g. the `C` and `gamma` of a
support vector machine) are passed to the estimator when it is
constructed.  The scikit‑learn documentation notes that it is
"possible and recommended" to search the hyper‑parameter space for the
best cross‑validation score【799207419018659†L125-L156】.  For each model we defined a
small parameter grid and used `GridSearchCV` with a 5‑fold
stratified split to identify the best combination.  Stratified splits
preserve the class distribution in each fold【722743218215345†L81-L90】, which is
important for medical datasets where one class may be under‑represented.

For logistic regression, the default `C=1.0` performed best and
grid search found no improvement; the model remains strong and
interpretable.  For random forests, the number of trees (`n_estimators`),
maximum tree depth and minimum split size were tuned.  For the SVM
with an RBF kernel, the penalty parameter `C` and kernel width `gamma`
were optimised.  For k‑NN, the number of neighbours (`k`), distance
metric (`p`) and weighting scheme were adjusted.  This tuning allows
each algorithm to perform competitively without manual trial‑and‑error.

### Model Evaluation

To estimate how each classifier generalises to unseen data, we used
5‑fold **stratified cross‑validation**.  Cross‑validation splits the
data into *k* folds; each fold is used once as a validation set while
the remaining `k−1` folds are used for training, and the resulting
scores are averaged.  Stratified k‑fold ensures that each fold
contains approximately the same proportion of malignant and benign
cases as the full dataset【722743218215345†L81-L90】.  This reduces the variance of
the performance estimate compared with a single train–test split and
prevents misleading results due to class imbalance.  The grid search
was driven by the mean cross‑validated accuracy, and the best
estimators were then evaluated on a separate 30 % hold‑out test set.
In addition to accuracy and ROC–AUC, we report precision, recall and
F1‑score, which are more sensitive to class‑specific performance.

A separate 30 % hold‑out test set was used for final evaluation.
Accuracy and ROC–AUC were computed, and confusion matrices and ROC
curves were plotted for each model.  These metrics provide insight into
both overall correctness and the ability to discriminate between
classes.

## Results
The tuned models show competitive performance, as summarised below
(values rounded to three decimal places).  Besides accuracy and
ROC–AUC, precision, recall and F1‑score are provided.  Higher values
indicate better performance.

| Model                   | CV Accuracy (mean) | Test Accuracy | Precision | Recall | F1‑score | ROC–AUC |
|-------------------------|--------------------|---------------|-----------|--------|----------|--------:|
| Logistic Regression     | 0.974              | **0.988**     | 0.991     | 0.991  | 0.991    | 0.998 |
| Random Forest           | 0.956              | 0.936         | 0.944     | 0.953  | 0.949    | 0.991 |
| SVM (RBF kernel)        | **0.981**          | 0.982         | 0.991     | 0.981  | 0.986    | 0.998 |
| k‑Nearest Neighbours    | 0.968              | 0.965         | 0.947     | **1.000** | 0.973    | 0.995 |

The models all achieve very high accuracy due to the separable nature
of the features.  The tuned support vector machine obtains the best
cross‑validated accuracy, while logistic regression remains the top
performer on the held‑out test set.  k‑NN achieves perfect recall
(i.e. all malignant tumours are correctly identified) at the cost of a
slight drop in precision.  Random forests benefit from tuning but
still lag behind the other models on this small dataset.

## Figures

For convenience, all the plots generated by the analysis are stored in
the `plots/` directory.  They include:

* **correlation_heatmap.png** – A 30×30 heatmap (lower triangle shown)
  depicting Pearson correlations between features.  By masking the
  upper triangle the plot removes redundant information and makes
  highly correlated clusters (e.g. radius, perimeter and area
  measures) easier to spot.
* **top_features_violin.png** – Violin plots comparing the six most
  predictive features across benign and malignant classes.  Colour is
  used consistently (blue for benign, red for malignant) and quartile
  markers illustrate the spread within each class.
* **confusion_matrix_*.png** – Separate confusion matrices for each
  tuned model.  The matrices are colour‑coded by percentage and
  annotated with raw counts to help explain trade‑offs between true
  positives (sensitivity) and true negatives (specificity).
* **roc_curve_*.png** – Individual ROC curves for each classifier
  plotted on their own axes, accompanied by a combined plot
  (**roc_curves_combined.png**) showing all four models together.  The
  area under the curve is reported in the legend.  A curve that hugs
  the top‑left corner indicates a model with strong discriminative
  ability.

These figures can be referenced in a technical discussion to
illustrate your analytical reasoning and communication skills.

## Usage

1. Clone the repository and navigate into it:

   ```bash
   git clone <repository‑url>
   cd breast_cancer_prediction
   ```

2. (Optional) Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   The main dependencies are `pandas`, `numpy`, `matplotlib`, `seaborn` and
   `scikit‑learn`.

3. Run the improved analysis script:

   ```bash
   python src/analysis_improved.py
   ```

   This will print summary statistics, perform hyperparameter tuning and
   cross‑validation, evaluate each model on a hold‑out test set, and
   save updated plots to `plots/`.

