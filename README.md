# MLITD Lab — Machine Learning for IT Application Development

This repository contains 10 lab experiments for the **Machine Learning for IT Application Development** course. Each experiment has its own folder with a Jupyter notebook and the required dataset.

## Folder Structure

```
25. ML-Experiments/
├── README.md                    ← This file
├── Exp_01/
│   ├── Exp_01.ipynb             ← Python Libraries for ML
│   └── student_placement.csv
├── Exp_02/
│   ├── Exp_02.ipynb             ← FIND-S Algorithm
│   └── play_tennis.csv
├── Exp_03/
│   ├── Exp_03.ipynb             ← Data Handling with Python
│   └── titanic_train.csv
├── Exp_04/
│   ├── Exp_04.ipynb             ← Linear & Multiple Linear Regression
│   └── house_train.csv
├── Exp_05/
│   ├── Exp_05.ipynb             ← Logistic Regression
│   └── titanic_train.csv
├── Exp_06/
│   ├── Exp_06.ipynb             ← K-Nearest Neighbour (KNN)
│   └── Iris.csv
├── Exp_07/
│   ├── Exp_07.ipynb             ← Feature Selection & Extraction
│   └── Iris.csv
├── Exp_08/
│   ├── Exp_08.ipynb             ← Support Vector Machine (SVM)
│   └── Iris.csv
├── Exp_09/
│   ├── Exp_09.ipynb             ← Decision Tree (CART)
│   └── Iris.csv
└── Exp_10/
    ├── Exp_10.ipynb             ← Deep Learning Case Study
    └── data.csv
```

## Experiment List

| # | Title | Dataset | CO |
|---|-------|---------|----|
| 01 | Exploring Python Libraries for ML | student_placement.csv | CO1 |
| 02 | FIND-S Algorithm | play_tennis.csv | CO1, CO2 |
| 03 | Data Handling with Python | titanic_train.csv | CO2 |
| 04 | Linear & Multiple Linear Regression | house_train.csv | CO2 |
| 05 | Logistic Regression | titanic_train.csv | CO3 |
| 06 | K-Nearest Neighbour (KNN) | Iris.csv | CO3, CO4 |
| 07 | Feature Selection & Extraction | Iris.csv | CO4 |
| 08 | Support Vector Machine (SVM) | Iris.csv | CO4 |
| 09 | Decision Tree (CART) | Iris.csv | CO4 |
| 10 | Deep Learning Case Study | data.csv | CO6 |

## Datasets

All CSV files are included in each experiment folder. Original sources:

| File | Source |
|------|--------|
| `student_placement.csv` | [Kaggle — Student Placement 2026](https://www.kaggle.com/datasets/sehaj1104/student-placement-prediction-dataset-2026) |
| `play_tennis.csv` | [Kaggle — Play Tennis Dataset](https://www.kaggle.com/code/devraai/play-tennis-dataset-classification-analysis) |
| `titanic_train.csv` | [Kaggle — Titanic](https://www.kaggle.com/competitions/titanic/data) |
| `house_train.csv` | [Kaggle — House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) |
| `Iris.csv` | [Kaggle — Iris Species](https://www.kaggle.com/uciml/iris/data) |
| `data.csv` | [Kaggle — Breast Cancer Wisconsin](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data) |

### Dataset Reuse
- **titanic_train.csv** is used by Exp_03 and Exp_05 (each folder has its own copy).
- **Iris.csv** is used by Exp_06, Exp_07, Exp_08, and Exp_09 (each folder has its own copy).

## Required Python Libraries

Install all dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow nltk scipy statsmodels
```

Or with conda:

```bash
conda install numpy pandas matplotlib seaborn scikit-learn nltk scipy statsmodels
pip install tensorflow
```

## How to Run

### Option 1: Jupyter Notebook (local)
```bash
cd Exp_01
jupyter notebook Exp_01.ipynb
```

### Option 2: Google Colab
1. Upload the experiment folder (notebook + CSV) to Google Drive.
2. Open the `.ipynb` file with Google Colab.
3. Upload the CSV to Colab's runtime or mount Google Drive.

### Option 3: VS Code
1. Open this folder in VS Code.
2. Install the Python and Jupyter extensions.
3. Open any `.ipynb` file and run cells interactively.

## Notes
- Each notebook reads its CSV from the **same folder** (e.g., `pd.read_csv('Iris.csv')`).
- All notebooks are self-contained — no cross-folder dependencies.
- Notebooks follow the structure: Aim → Theory → Procedure → Output → Conclusion.
- No viva questions are included.
