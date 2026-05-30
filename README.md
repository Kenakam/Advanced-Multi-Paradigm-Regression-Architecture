# Advanced Multi-Paradigm Regression Architecture: Predictive Modeling & Deep Neural Optimization for Deal Probability Forecasting

## 📌 Project Overview

This repository houses the final capstone project for the Advanced Regression Analysis course—a comprehensive, multi-paradigm machine learning and deep neural framework designed to predict online advertisement **deal probability**. Utilizing an intricate multi-modal and highly sparse data asset derived from the **Avito Demand Prediction** ecosystem, the objective is to model the continuous likelihood ($[0, 1]$) of a user transaction based on structural, text-based, and financial advertisement metrics.

To maximize predictive signals and uncover non-linear relationships, this repository implements two independent pipelines featuring **more than 20 distinct predictive model variations**. These configurations span across traditional regularized estimators, distance-based mappings, high-dimensional Principal Component Analysis (PCA) pipelines, advanced Gradient Boosted Decision Trees (GBDTs), automated **Optuna** Bayesian optimization frameworks, complex multi-layer Artificial Neural Networks (ANNs), and custom Meta-Learning Ensemble architectures (Voting and Stacking Regressors).

---

## 🛠️ Unified Data Engineering & Preprocessing Pipeline

Before splitting into classical machine learning and neural architectures, the source data undergoes a rigorous, production-grade feature engineering and numerical sanitization pipeline to clean and structuralize highly noisy user interactions.



```
              [ Raw Data Asset Ingestion ]
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
 [ Textual Analytics ]           [ Financial & Temporal ]
 • Title/Desc Character Length    • Logarithmic Pricing
 • Word Counts & Ratio Vectors    • Category Relative Price Ratio
           └───────────────┬───────────────┘
                           ▼
           [ Missing Value Imputation ]
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
   [ Tree-Based Split ]            [ Distance/Linear Split ]
   • Dense Raw Features            • Standard Scaling (Z-score)
   • Native Categorical Arrays     • Principal Component Analysis (PCA)

```



### 1. Advanced Feature Extraction
* **Temporal Dynamics**: Extracted `activation_weekday` and a specialized boolean flag `is_weekend` from raw activation dates to capture periodic variations in consumer browsing traffic.
* **Textual Analytics**: Generated `title_len`, `description_len`, and `desc_word_count` to map structural listing profiles, paired with a custom `title_desc_len_ratio` vector to quantify descriptive density balance.
* **Financial Scaling & Relational Ratios**: Addressed extreme variance and severe right-skewness in currency metrics by calculating `price_log`. Developed localized structural metrics, including `price_rel_to_cat` (individual price divided by its structural category mean) and interaction vectors such as `price_to_seq_ratio`.
* **Behavioral Aggregation**: Engineered `user_activity_count` to capture historical repetition and user posting velocity.

### 2. Missing Value Imputation & Encoding
* Missing elements across dense numerical attributes were resolved using localized median replacement. 
* High-cardinality categorical vectors (`region`, `city`, `parent_category_name`, `category_name`, `user_type`, and operational `params`) were processed using label and ordinal transformations into dense arrays.

### 3. Dimensionality Reduction & Multicollinearity Filtering
* Multicollinearity matrices were audited via Variance Inflation Factors (VIF) to detect feature redundancies. To optimize learning rates across different model types:
  * **Tree-Based & Boosting Models**: Retained the dense tabular structure to leverage native decision path split mechanics.
  * **Distance & Linear Models (e.g., KNN)**: Channeled through standard scaling ($Z$-score normalization) followed by **Principal Component Analysis (PCA)** to transform high-dimensional, multi-collinear spaces into orthogonal, high-variance components.

---

## 📊 Section 1: Classical Machine Learning & Tree-Based Frameworks
*Notebook Target: `(ML) Advanced Multi-Paradigm Regression.ipynb`*

This pipeline builds and evaluates over 20 distinct model-data configurations, measuring performance on both raw engineering matrices and compressed orthogonal PCA spaces.

### 📦 Structural Framework Portfolio

#### 1. Baseline Estimators & Linear Variants
* **Ordinary Least Squares (OLS) Regression**: Established a foundational benchmark for linear target boundaries.
* **PCA-Driven Linear Regressors**: Built models on top of low-rank orthogonal matrices to eliminate structural multicollinearity.

#### 2. Instance-Based & Non-Parametric Mappings
* **K-Nearest Neighbors (KNN) Regressor**: Evaluated across multi-dimensional distance metrics and hyperparameter clusters. The model relied heavily on the low-rank PCA space to avoid distance metric degradation caused by the "curse of dimensionality."

#### 3. Bagging & Ensemble Tree Configurations
* **Random Forest Regressor**: Configured with extensive tree depth parameters to evaluate randomized feature subsets.

#### 4. Advanced Gradient Boosted Decision Trees (GBDT)
* **XGBoost (eXtreme Gradient Boosting)**: Fine-tuned with precise regularization weights (`alpha`, `lambda`) and custom subsampling frequencies to manage residual corrections.
* **LightGBM**: Utilized leaf-wise growth configurations (`num_leaves`, `max_depth`) to capture categorical splits with minimal training overhead.
* **CatBoost (Categorical Boosting)**: Deployed to natively handle residual mappings across structural categorical arrays.

### 🎛️ Hyperparameter Optimization & Explainability
* **Automated Tuning via Optuna**: Integrated automated hyperparameter tuning driven by a Tree-structured Parzen Estimator (TPE) sampler to systematically discover optimal leaf structures, learning rate schedules, and tree frequencies.
* **Model Interpretation via SHAP**: Deployed **SHAP (SHapley Additive exPlanations)** to break down individual split parameters and visualize feature impact across gradient boosting decision paths.

### 🚀 Meta-Learning Ensemble Topologies
To extract maximum predictive power from the classical framework, top-tier estimators were combined into meta-ensembles:
* **Voting Regressor**: Blended the continuous predictive distributions of individual high-performing boosting models.
* **Stacking Regressor**: Combined individual base estimators underneath a final regularized meta-estimator to learn patterns within cross-model residual predictions ($R^2_{\text{Train}}: 0.3179 \rightarrow R^2_{\text{Test}}: 0.1788$).

---

## 🧠 Section 2: Deep Learning & Artificial Neural Networks (ANN)
*Notebook Target: `(DL) Advanced Multi-Paradigm Regression.ipynb`*

This notebook develops a fully deep learning architecture designed to map complex, non-linear relationships within a continuous target boundary ($[0, 1]$).



```
                 [ Scaled Continuous Inputs ]
                              │
                              ▼
                 [ Dense Input Layer (ReLU) ]
                              │
                              ▼
           ┌──────────────────────────────────┐
           │    Optuna Dynamic Hidden Layers  │
           │   ┌──────────────────────────┐   │
           │   │   Dense Layer (16-512)   │   │
           │   └─────────────┬────────────┘   │
           │                 ▼                │
           │   ┌──────────────────────────┐   │
           │   │   Dropout Regularization │   │
           │   └──────────────────────────┘   │
           └─────────────────┬────────────────┘
                             │ (Iterated 1 to 5 Times)
                             ▼
                 [ Output Layer (Linear) ]


```

### 📐 Neural Architecture & Layer Design
The deep network converts a standardized input array into a highly regularized hidden architecture before converging into a single regression node:
* **Input Standardization**: Processed through robust Scikit-Learn scaling layers to maintain stable weight gradients during backpropagation.
* **Hidden Layers**: Built with alternating sequences of dense operations combined with `Dropout` layers to prevent parameter co-adaptation.
* **Activation Nodes**: Implemented Rectified Linear Unit (`ReLU`) variants to maintain stable gradients throughout backpropagation.

### 🎛️ Optuna-Driven Neural Space Searching
Optuna was integrated into the training pipeline to dynamically test and assemble neural network structures. The optimization search space covered:
* **Layer Scaling**: Dynamically expanded network depth between 1 and 5 hidden dense layers.
* **Neuron Topographies**: Swept individual layer configurations from 16 hidden units up to 512 units.
* **Dropout Densities**: Varied dropout ratios between `0.0` and `0.5` to manage generalization error.
* **Optimization Algorithms**: Evaluated performance across several backpropagation optimization algorithms, including `SGD`, `Adam`, `RMSprop`, `Adagrad`, `Adadelta`, and `Nadam`.
* **Learning Rate Curves**: Sampled exponentially across log-space scales between `1e-5` and `1e-1`.

---

## ⚙️ Operational Thresholds & Hardware Constraints

 ### ⚠️ Task-Related Operational Thresholds
>Certain hyperparameters, row splits, token dimensions, and PCA component variances utilized within these scripts deviate from traditional industry production configurations. **These specific boundaries represent task-related constraints explicitly aligned with the grading requirements and target boundaries established for this course project.** These limitations ensure structured analytical consistency across all peer evaluations.
>
>

### 🖥️ Hardware & Resource Performance Realities
>All model training, grid searches, Optuna iterations, and dense neural weight updates were carried out entirely within a **local machine** and standard **Google Colab** environments. Because these platforms operate under restricted local GPU/CPU allocation matrices and runtime execution timers, the capacity to perform unconstrained hyperparameter trials, massive cross-validation splits, or uninhibited epoch counts was physically bounded. Additionally, due to potential memory constraints, the machine learning and deep learning models have been separated into two distinct notebooks.
>
>

>Consequently, model performance metrics (such as the final $R^2$ scores) can present a "not ideal" convergence output compared to production systems. In an unconstrained infrastructure environment backed by enterprise-grade distributed compute clusters, running significantly extended trial budgets and expanded neural architectures would yield lowered error metrics and more optimal tracking profiles.
>
>

---

## 💻 Technical Stack Matrix

* **Languages & Core Ecosystems**: Python 3.x, Jupyter Workspace, Google Colab
* **Data Processing & Transformations**: Pandas, NumPy, OpenPyXL, Scikit-Learn (`StandardScaler`, `LabelEncoder`, `PCA`)
* **Machine Learning Engines**: Scikit-Learn, CatBoost, XGBoost, LightGBM
* **Deep Learning Infrastructure**: TensorFlow, Keras API (Sequential & Functional)
* **Hyperparameter Optimization Suite**: Optuna Framework (`TPESampler`)
* **Model Explainability & Insights**: SHAP (SHapley Additive exPlanations)
* **Visualization Suite**: Matplotlib, Seaborn

---

## 🚀 Deployment & Replication Steps

### 1. Workspace Organization
To ensure smooth execution, verify that your local workspace includes the following files in its root folder:

```

.
├── regression_avito_deals.xlsx                         # Source Excel Data Asset File
├── (ML) Advanced Multi-Paradigm Regression.ipynb       # Classical ML, Boosting & Ensemble Pipeline
└── (DL) Advanced Multi-Paradigm Regression.ipynb       # Deep Learning & Optuna Neural Optimizer

```

### 2. Environment Installation
Install the necessary library versions using your terminal or a Jupyter notebook cell:
```bash
pip install numpy pandas scikit-learn tensorflow optuna catboost xgboost lightgbm shap openpyxl matplotlib seaborn

```

### 3. Pipeline Ingestion Protocol

* **Step 1**: Execute `(ML) Advanced Multi-Paradigm Regression.ipynb` to complete classical feature engineering, apply PCA transformations, and evaluate the gradient boosting and ensemble benchmarks.
* **Step 2**: Execute `(DL) Advanced Multi-Paradigm Regression.ipynb` to process the standardized data arrays and initiate the automated Optuna neural network optimization cycles.


## NOTE: Because the file size is too large, I'm providing a link for it:
* regression_avito_deals.xlsx: [https://drive.google.com/file/d/1y5R514fgg2_0CQ3CZhpaW13Tn5TnUWPP/view?usp=drive_link](https://docs.google.com/spreadsheets/d/1V4aovw43J-B-_LHXREO8jXvB9vN_pM9f/edit?usp=sharing&ouid=110983974899597192480&rtpof=true&sd=true)
