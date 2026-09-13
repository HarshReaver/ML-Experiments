# Machine Learning Practical Implementation and Notes

This document contains short, functional Python programs for the Machine Learning practical experiments. The code is kept simple so that it can be understood and written during a lab session, while still covering the main points given in the lab manual.

---

# Experiment 1: Case study on different Python tools and libraries used in ML

## Aim

To explore and understand key Python libraries commonly used in machine learning, with practical examples demonstrating their usage.

## Dataset

Use a student placement CSV containing the fields required by the exercises below.

Suggested file name:

`student_placement.csv`

The original manual uses fields such as `DegreePercentage`, `Salary`, `Gender`, `Branch`, `WorkExperience`, `Placed`, `Company`, and `Comments`.

## 1. NumPy

NumPy is used for numerical calculations and array operations.

### Exercise

Calculate mean Degree Percentage and median Salary.

```python
import numpy as np

degree = df["DegreePercentage"].to_numpy()
salary = df["Salary"].to_numpy()

print("Mean Degree Percentage:", np.mean(degree))
print("Median Salary:", np.median(salary))
```

### Code Explanation

- `to_numpy()` converts a Pandas column into a NumPy array.
- `np.mean()` finds the average.
- `np.median()` finds the middle value.

---

## 2. Pandas

Pandas is used for reading, filtering and manipulating tabular data.

### Exercise

Preview the data and filter placed female students with DegreePercentage greater than 75.

```python
import pandas as pd

print(df.head())

placed_females = df[
    (df["Placed"] == "Yes") &
    (df["Gender"] == "Female") &
    (df["DegreePercentage"] > 75)
]

print(placed_females[
    ["StudentID", "Gender", "DegreePercentage", "Company"]
])
```

### Code Explanation

- `head()` shows the first few rows.
- Boolean conditions filter the required students.
- Pandas makes tabular filtering simple.

---

## 3. Matplotlib

Matplotlib is used for basic graphs and charts.

### Exercise

Plot the salary distribution of placed students.

```python
import matplotlib.pyplot as plt

salary = df[df["Placed"] == "Yes"]["Salary"]

plt.hist(salary, bins=15)
plt.title("Salary Distribution of Placed Students")
plt.xlabel("Salary")
plt.ylabel("Number of Students")
plt.show()
```

### Code Explanation

A histogram shows how the salary values are distributed.

---

## 4. Seaborn

Seaborn is a high-level visualization library built on Matplotlib.

### Exercise

Visualize placement count by branch.

```python
import seaborn as sns

sns.countplot(x="Branch", hue="Placed", data=df)
plt.title("Placement Count by Branch")
plt.xticks(rotation=45)
plt.show()
```

### Code Explanation

`countplot()` counts the number of records in each category and separates them using the placement status.

---

## 5. Scikit-learn

Scikit-learn provides ready-to-use machine learning algorithms.

### Exercise

Build and evaluate a Logistic Regression model to predict placement.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = df.copy()

for col in ["Gender", "Branch", "WorkExperience", "Placed"]:
    data[col] = LabelEncoder().fit_transform(data[col])

X = data[[
    "Gender", "Branch", "DegreePercentage", "WorkExperience"
]]
y = data["Placed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, y_pred))
```

### Code Explanation

The categorical columns are converted into numbers, then the data is split into training and testing sets. Logistic Regression is trained on the training data and tested using accuracy.

---

## 6. TensorFlow / Keras

TensorFlow is used for machine learning and deep learning. Keras provides a simpler interface for creating neural networks.

### Exercise

Train a simple neural network to predict placement.

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=1
)

model = Sequential([
    Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=20, verbose=0)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("Test accuracy:", accuracy)
```

### Code Explanation

- `Dense` creates neural network layers.
- `relu` is used in hidden layers.
- `sigmoid` gives a binary output.
- `adam` is used for optimization.
- The model is trained for a small number of epochs to keep the practical short.

---

## 7. NLTK

NLTK is used for processing human language and text.

### Exercise

Tokenize a sample student comment.

```python
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")

sample_comment = df.loc[0, "Comments"]
tokens = word_tokenize(sample_comment)

print("Original comment:", sample_comment)
print("Tokens:", tokens)
```

### Code Explanation

Tokenization breaks a sentence into smaller units such as words and punctuation.

---

## 8. SciPy

SciPy provides scientific and statistical functions.

### Exercise

Perform a t-test to compare DegreePercentage of placed and not placed students.

```python
from scipy import stats

placed = df[df["Placed"] == "Yes"]["DegreePercentage"]
not_placed = df[df["Placed"] == "No"]["DegreePercentage"]

t_stat, p_val = stats.ttest_ind(placed, not_placed)

print("T-statistic:", round(t_stat, 3))
print("P-value:", round(p_val, 3))
```

### Code Explanation

The t-test checks whether the two groups have a statistically meaningful difference in their degree percentages.

---

## 9. Statsmodels

Statsmodels is useful when a detailed statistical model summary is required.

### Exercise

Get a detailed Logistic Regression summary report.

```python
import statsmodels.api as sm

X_sm = data[[
    "Gender", "Branch", "DegreePercentage", "WorkExperience"
]]

X_sm = sm.add_constant(X_sm)
y_sm = data["Placed"]

model_sm = sm.Logit(y_sm, X_sm)
result = model_sm.fit()

print(result.summary())
```

### Code Explanation

Unlike the basic scikit-learn accuracy output, Statsmodels provides a detailed statistical summary containing coefficients, standard errors, p-values and other information.

## Conclusion

The experiment gives a basic working understanding of NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras, NLTK, SciPy and Statsmodels. Together these libraries cover numerical work, data handling, visualization, machine learning, deep learning, text processing and statistical analysis.

---

# Experiment 2: FIND-S and Candidate-Elimination algorithm on training data from CSV

## Aim

Write a code to the FIND-S algorithm for finding the most specific hypothesis based on a given set of training data samples. Read the training data from a `.CSV` file.

## Dataset

Use the small **EnjoySport** training dataset from the lab manual.

Suggested file name:

`training_data.csv`

```text
Sky,AirTemp,Humidity,Wind,Water,Forecast,EnjoySport
Sunny,Warm,Normal,Strong,Warm,Same,Yes
Sunny,Warm,High,Strong,Warm,Same,Yes
Rainy,Cold,High,Strong,Warm,Change,No
Sunny,Warm,High,Strong,Cool,Change,Yes
```

## FIND-S: Code Implementation

```python
import pandas as pd

df = pd.read_csv("training_data.csv")

attributes = df.columns[:-1]
target = df.columns[-1]

hypothesis = ["0"] * len(attributes)

for _, row in df.iterrows():
    if row[target].strip().lower() == "yes":

        if hypothesis == ["0"] * len(attributes):
            hypothesis = list(row[attributes])

        else:
            for i in range(len(attributes)):
                if hypothesis[i] != row[attributes[i]]:
                    hypothesis[i] = "?"

print("Final Hypothesis:", hypothesis)
```

## Code Explanation

- `0` represents the initial most specific hypothesis.
- Only positive examples are used.
- When two positive examples have different values for an attribute, that attribute becomes `?`.
- `?` means any value is acceptable.

For the EnjoySport example, the final hypothesis is:

```text
<Sunny, Warm, ?, Strong, ?, ?>
```

## Candidate-Elimination: Simplified Implementation

The lab manual introduces Candidate Elimination along with FIND-S. Candidate Elimination keeps two boundaries:

- `S`: the most specific consistent hypotheses.
- `G`: the most general consistent hypotheses.

A compact implementation for the EnjoySport data is shown below.

```python
import pandas as pd

df = pd.read_csv("training_data.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].str.lower().values

n = X.shape[1]

# Start with the most specific and most general boundaries
S = ["0"] * n
G = [["?"] * n]

for row, target in zip(X, y):

    if target == "yes":
        # Generalize S for a positive example
        if S == ["0"] * n:
            S = list(row)
        else:
            for i in range(n):
                if S[i] != row[i]:
                    S[i] = "?"

        # Remove general hypotheses that do not cover S
        G = [
            g for g in G
            if all(g[i] == "?" or g[i] == S[i] for i in range(n))
        ]

    else:
        # Specialize G for a negative example
        new_G = []

        for g in G:
            for i in range(n):
                if g[i] == "?" and S[i] != "?" and S[i] != row[i]:
                    g_new = g.copy()
                    g_new[i] = S[i]
                    new_G.append(g_new)

        if new_G:
            G = new_G

print("Specific Boundary S:", S)
print("General Boundary G:", G)
```

### Code Explanation

- A positive example makes `S` more general where required.
- A negative example causes `G` to be specialized.
- `S` and `G` together represent the current version space boundary.
- For the lab record, the FIND-S result remains the main required output; the Candidate-Elimination section demonstrates the extension mentioned in the experiment title.

## Conclusion

FIND-S finds the most specific hypothesis consistent with positive examples. Candidate Elimination extends this idea by keeping both specific and general boundaries.

---

# Experiment 3: Data Handling with Python

## Aim

To understand and perform data handling using Python libraries (NumPy and Pandas) for preprocessing datasets in Machine Learning.

## Dataset

**Titanic Dataset** from Kaggle.

Suggested file name:

`titanic_train.csv`

## Code Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("titanic_train.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nSummary:")
print(df.describe())

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Fill missing Age values
df["Age"] = df["Age"].fillna(df["Age"].median())

# Remove duplicates
df = df.drop_duplicates()

# Encode categorical data
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Create new feature
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Descriptive statistics
print("\nAge statistics:")
print(
    df["Age"].mean(),
    df["Age"].median(),
    df["Age"].std()
)

# Visualization
sns.histplot(df["Age"], kde=True)
plt.title("Age Distribution")
plt.show()

sns.countplot(x="Sex", data=df)
plt.title("Gender Count")
plt.show()

# Save cleaned data
df.to_csv("cleaned_titanic.csv", index=False)
```

## Code Explanation

- `read_csv()` loads the dataset.
- `head()`, `info()` and `describe()` are used to understand the data.
- `isnull()` checks missing values.
- `fillna()` fills missing Age values.
- `drop_duplicates()` removes duplicate rows.
- `map()` converts categorical values to numerical values.
- `FamilySize` is a new feature.
- Matplotlib and Seaborn are used for visualization.
- The cleaned data is saved for later ML tasks.

## Result

Data was loaded, explored, cleaned, transformed and visualized using Python.

---

# Experiment 4: Linear and Multiple Linear Regression

## Aim

To implement Linear Regression and Multiple Linear Regression using Python and evaluate their performance on a given dataset.

## Dataset

**Diamonds dataset** may be used as the simple demonstration dataset from the manual.

Suggested file name:

`diamonds.csv`

For a purely Kaggle-based workflow, the same regression structure can also be applied to the **House Prices - Advanced Regression Techniques** dataset.

## Code Implementation

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("diamonds.csv")

# Simple Linear Regression
X = data[["carat"]]
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Simple Linear Regression")
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.title("Simple Linear Regression")
plt.xlabel("Carat")
plt.ylabel("Price")
plt.show()

# Multiple Linear Regression
X_multi = data[["carat", "depth", "table"]]

X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y, test_size=0.2, random_state=42
)

multi_model = LinearRegression()
multi_model.fit(X_train, y_train)

y_pred_multi = multi_model.predict(X_test)

print("\nMultiple Linear Regression")
print("Coefficients:", multi_model.coef_)
print("Intercept:", multi_model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred_multi))
print("R2 Score:", r2_score(y_test, y_pred_multi))
```

## Code Explanation

- Simple Linear Regression uses one independent variable.
- Multiple Linear Regression uses more than one independent variable.
- `fit()` trains the model.
- `predict()` produces predicted values.
- MSE measures prediction error.
- R² measures how well the model fits the data.

## Result

Both simple and multiple linear regression were implemented and evaluated using MSE and R².

---

# Experiment 5: Logistic Regression

## Aim

To implement Logistic Regression for binary classification using Python and evaluate its performance.

## Dataset

**Titanic Dataset** from Kaggle.

Suggested file name:

`titanic_train.csv`

## Code Implementation

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

df = pd.read_csv("titanic_train.csv")

df = df[["Age", "Fare", "Sex", "Survived"]].dropna()

df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})

X = df[["Age", "Fare", "Sex"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

# Probability prediction
y_prob = model.predict_proba(X_test)[:, 1]

print("\nFirst five probabilities:")
print(y_prob[:5])
```

## Code Explanation

- Logistic Regression is used for binary classification.
- `Survived` is the target variable.
- The confusion matrix shows correct and incorrect predictions.
- The classification report gives precision, recall and F1-score.
- `predict_proba()` gives the probability of class 1.

## Result

Logistic Regression was implemented for binary classification and evaluated using standard classification metrics.

---

# Experiment 6: K-Nearest Neighbour (KNN) Algorithm

## Aim

To implement and demonstrate the working of the K-Nearest Neighbour (KNN) algorithm for classification using Python.

## Dataset

**Iris Dataset** from Kaggle.

Suggested file name:

`Iris.csv`

## Code Implementation

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

df = pd.read_csv("Iris.csv")

X = df[
    ["SepalLengthCm", "SepalWidthCm",
     "PetalLengthCm", "PetalWidthCm"]
]

y = df["Species"]

# Feature scaling
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
```

## Decision Boundary Visualization

```python
from matplotlib.colors import ListedColormap
import numpy as np

X_2d = df[
    ["SepalLengthCm", "SepalWidthCm"]
].values

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_2d, y, test_size=0.2, random_state=42
)

knn2 = KNeighborsClassifier(n_neighbors=3)
knn2.fit(X_train2, y_train2)

x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.1),
    np.arange(y_min, y_max, 0.1)
)

Z = knn2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=pd.factorize(y)[0])

plt.title("KNN Decision Boundary (K=3)")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()
```

## Code Explanation

KNN classifies a new data point using the majority class among its nearest neighbours. Here `K=3`, so three nearby points are considered. Scaling is important because KNN is distance-based.

## Result

KNN was implemented on the Iris dataset and evaluated using a confusion matrix, classification report and accuracy.

---

# Experiment 7: Feature Selection and Feature Extraction Techniques

## Aim

To implement and understand feature selection and feature extraction techniques for dimensionality reduction and improving model performance in Machine Learning.

## Dataset

**Iris Dataset** from Kaggle.

Suggested file name:

`Iris.csv`

## Code Implementation

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("Iris.csv")

features = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]

X = df[features]
y = df["Species"]

print(X.head())

# 1. Chi-Square Feature Selection
selector = SelectKBest(score_func=chi2, k=2)
X_new = selector.fit_transform(X, y)

print("Selected feature indices:",
      selector.get_support(indices=True))

# 2. RFE
model = LogisticRegression(max_iter=200)
rfe = RFE(model, n_features_to_select=2)
rfe.fit(X, y)

print("RFE selected features:", rfe.support_)
print("Feature ranking:", rfe.ranking_)

# 3. PCA Feature Extraction
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:",
      pca.explained_variance_ratio_)

plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("PCA Feature Extraction (2D)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
```

## Code Explanation

- **Chi-Square** is a filter method.
- **RFE** is a wrapper method.
- **PCA** is a feature extraction method.
- Feature selection keeps original features.
- Feature extraction creates new transformed features.
- PCA is used to reduce four dimensions to two.

## Result

Relevant features were selected using Chi-Square and RFE, while PCA was used to create a lower-dimensional representation.

---

# Experiment 8: Support Vector Machine (SVM) Algorithm

## Aim

To implement and demonstrate the working of Support Vector Machine (SVM) algorithm for classification using Python.

## Dataset

**Iris Dataset** from Kaggle.

Suggested file name:

`Iris.csv`

## Code Implementation

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

df = pd.read_csv("Iris.csv")

X = df[
    ["SepalLengthCm", "SepalWidthCm",
     "PetalLengthCm", "PetalWidthCm"]
]

y = df["Species"]

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear kernel
model_linear = SVC(kernel="linear")
model_linear.fit(X_train, y_train)

y_pred_linear = model_linear.predict(X_test)

print("Linear SVM")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_linear))
print("Accuracy:",
      accuracy_score(y_test, y_pred_linear))

# RBF kernel
model_rbf = SVC(kernel="rbf", gamma="auto")
model_rbf.fit(X_train, y_train)

y_pred_rbf = model_rbf.predict(X_test)

print("\nRBF SVM")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rbf))
print("Accuracy:",
      accuracy_score(y_test, y_pred_rbf))
```

## Decision Boundary Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

X_2d = df[
    ["SepalLengthCm", "SepalWidthCm"]
].values

y_2d = pd.factorize(y)[0]

svc = SVC(kernel="linear")
svc.fit(X_2d, y_2d)

x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("SVM Decision Boundary (Linear Kernel)")
plt.show()
```

## Code Explanation

- SVM finds a hyperplane with the maximum margin between classes.
- Support vectors are the points closest to the boundary.
- The linear kernel works with a linear decision boundary.
- The RBF kernel can model non-linear boundaries.
- `C` controls the penalty for classification errors.
- `gamma` controls the influence of individual points in RBF.

## Result

SVM was implemented using both linear and RBF kernels, and the models were evaluated using classification metrics.

---

# Experiment 9: Q-learning, a fundamental reinforcement learning algorithm

## Note about the lab manual

The experiment list at the beginning of the supplied lab manual lists **Q-learning** as Experiment 9, while the detailed pages later in the same manual describe **Decision Tree (CART)** as Experiment 9. Both topics are therefore included here so that the practical notes cover the material present in the supplied manual.

## Aim / Task

Implement Q-learning, a fundamental reinforcement learning algorithm.

## Dataset

No external dataset is required. A small state-based environment is enough to understand the algorithm.

## Code Implementation

```python
import numpy as np

# States: 0 -> 1 -> 2 -> 3 (goal)
# Actions: 0 = left, 1 = right

Q = np.zeros((4, 2))

alpha = 0.8      # learning rate
gamma = 0.9      # discount factor
episodes = 100

for _ in range(episodes):
    state = 0

    while state != 3:
        action = np.random.choice([0, 1])

        if action == 1 and state < 3:
            next_state = state + 1
        elif action == 0 and state > 0:
            next_state = state - 1
        else:
            next_state = state

        reward = 10 if next_state == 3 else -1

        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state])
            - Q[state, action]
        )

        state = next_state

print("Q-Table:")
print(np.round(Q, 2))

print("Best action for each state:",
      np.argmax(Q, axis=1))
```

## Code Explanation

- The agent starts in state `0`.
- It chooses an action and observes the next state and reward.
- The Q-table stores the learned value of each state-action pair.
- `alpha` controls how quickly new information is learned.
- `gamma` controls the importance of future rewards.
- Repeating the process over many episodes helps the agent learn the best action.

## Result

The Q-table is updated through repeated interaction with the environment, and the highest-valued action is selected for each state.

---

# Experiment 9: Decision Tree (CART) Algorithm

## Aim

To implement and demonstrate the working of the Decision Tree (CART) algorithm for classification using Python.

## Dataset

**Iris Dataset** from Kaggle.

Suggested file name:

`Iris.csv`

## Code Implementation

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

df = pd.read_csv("Iris.csv")

X = df[
    ["SepalLengthCm", "SepalWidthCm",
     "PetalLengthCm", "PetalWidthCm"]
]

y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CART using Gini Index
clf_gini = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

clf_gini.fit(X_train, y_train)

y_pred_gini = clf_gini.predict(X_test)

print("Gini Index")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_gini))
print("Accuracy:",
      accuracy_score(y_test, y_pred_gini))

# Decision tree visualization
plt.figure(figsize=(12, 8))
plot_tree(
    clf_gini,
    feature_names=X.columns,
    class_names=clf_gini.classes_,
    filled=True
)

plt.title("Decision Tree (CART) using Gini Index")
plt.show()

# Entropy
clf_entropy = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)

clf_entropy.fit(X_train, y_train)
y_pred_entropy = clf_entropy.predict(X_test)

print("\nAccuracy (Entropy):",
      accuracy_score(y_test, y_pred_entropy))
```

## Code Explanation

- CART stands for Classification and Regression Trees.
- Gini and Entropy are used as splitting criteria.
- `max_depth=3` limits the tree size.
- Limiting depth helps reduce overfitting.
- `plot_tree()` displays the learned tree.

## Result

Decision Trees using Gini Index and Entropy were implemented and compared.

---

# Experiment 10: Case Study on Deep Learning: Comprehensive Analysis and Implementation

## Case Study

Healthcare - Medical Image Analysis for Cancer Detection.

The manual uses the Wisconsin Breast Cancer dataset as the underlying dataset and demonstrates a neural-network based medical diagnosis case study.

## Dataset

Use the **Breast Cancer Wisconsin (Diagnostic)** dataset.

For a simple practical, the dataset can be loaded using scikit-learn, which avoids unnecessary image-processing setup:

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset shape:", X.shape)
print("Classes:", data.target_names)
```

## Required Libraries

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

## Dataset and Preprocessing

```python
data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
```

## Deep Learning Model

```python
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()
```

## Train the Model

```python
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
```

## Model Evaluation and Results

```python
test_loss, test_accuracy = model.evaluate(
    X_test, y_test, verbose=0
)

y_pred = (
    model.predict(X_test) > 0.5
).astype(int).ravel()

print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names
    )
)
```

## Training History

```python
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("Deep Learning Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.show()
```

## Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=data.target_names,
    yticklabels=data.target_names
)

plt.title("Medical Diagnosis Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

## Code Explanation

- The dataset is split into training and testing data.
- StandardScaler brings the numerical features to a comparable scale.
- The network contains several dense layers, batch normalization and dropout.
- ReLU is used in the hidden layers.
- Sigmoid is used for binary classification.
- Adam is used as the optimizer.
- Accuracy and loss are recorded during training.
- A confusion matrix and classification report are used for evaluation.

## Important Note

The original manual describes the case as medical image analysis and also contains a CNN example. The actual code in the manual converts the tabular Wisconsin breast-cancer features into an image-like shape and then provides a Dense Neural Network alternative. For a short and reliable college practical, the Dense Neural Network above is the cleaner implementation while preserving the same cancer-detection case study.

## Conclusion

The deep learning case study demonstrates how a neural network can be trained for a binary medical classification problem and evaluated using accuracy, loss, a classification report and a confusion matrix.

---

# Dataset Reference

## 1. Student Placement Prediction Dataset 2026

Kaggle:

https://www.kaggle.com/datasets/sehaj1104/student-placement-prediction-dataset-2026

Used for:
- Experiment 1

## 2. Titanic - Machine Learning from Disaster

Kaggle:

https://www.kaggle.com/c/titanic

Used for:
- Experiment 3
- Experiment 5

## 3. Iris Species

Kaggle:

https://www.kaggle.com/uciml/iris

Used for:
- Experiment 6
- Experiment 7
- Experiment 8
- Experiment 9

## 4. House Prices - Advanced Regression Techniques

Kaggle:

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

Can be used for:
- Experiment 4

## 5. Breast Cancer Wisconsin (Diagnostic)

Kaggle:

https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

Used for:
- Experiment 10

---

# Viva Questions and Answers

## Experiment 1

### 1. Name any four Python libraries used in Machine Learning.

NumPy, Pandas, Matplotlib and Scikit-learn are four commonly used libraries.

### 2. What is the primary use of NumPy in ML?

NumPy is mainly used for numerical calculations, arrays, matrices and mathematical operations.

### 3. How does Pandas help in data preprocessing?

Pandas helps load datasets, handle missing values, filter rows, modify columns, encode data and save cleaned datasets.

### 4. Which library is commonly used for data visualization in ML?

Matplotlib is a commonly used visualization library. Seaborn is another library used for statistical plots.

### 5. What is Scikit-learn mainly used for?

Scikit-learn provides machine learning algorithms for classification, regression, clustering, preprocessing and model evaluation.

---

## Experiment 2

### 1. What is the purpose of the FIND-S algorithm?

FIND-S finds the most specific hypothesis that is consistent with the positive training examples.

### 2. Why does FIND-S consider only positive training examples?

FIND-S is designed to generalize the hypothesis using positive examples. Negative examples are ignored.

### 3. What is meant by the most specific hypothesis?

It is the hypothesis that makes the fewest general assumptions while still covering the positive examples.

### 4. Can FIND-S handle noisy data? Why or why not?

Not very well. FIND-S is sensitive to noisy or incorrect positive examples because each positive example can change the hypothesis.

### 5. What is the difference between a specific and general hypothesis?

A specific hypothesis accepts fewer cases, while a general hypothesis accepts more possible cases.

---

## Experiment 3

### 1. What are the advantages of using Pandas over Python lists for data handling?

Pandas provides DataFrames, filtering, missing-value handling, grouping, statistics and CSV operations directly.

### 2. Explain the difference between NaN and None in Python.

`NaN` is a special floating-point value used to represent missing numerical data. `None` is Python's null object.

### 3. How do you handle missing values in datasets? Name two strategies.

Two common methods are filling missing values with a statistic such as mean or median, and removing rows or columns with too many missing values.

### 4. What is the purpose of `describe()` and `info()` in Pandas?

`info()` gives column names, data types and non-null counts. `describe()` gives statistical summaries such as mean, standard deviation, minimum and maximum.

### 5. Why is data cleaning important in Machine Learning?

Dirty data can reduce model accuracy and may lead to misleading results.

---

## Experiment 4

### 1. What is the difference between simple and multiple linear regression?

Simple linear regression uses one independent variable. Multiple linear regression uses two or more independent variables.

### 2. What assumptions are made in linear regression?

Common assumptions include linearity, independent observations, constant error variance and errors that are approximately normally distributed.

### 3. What do coefficients and intercept represent in the regression equation?

The intercept is the predicted value when the input features are zero. A coefficient shows how much the predicted output changes for a one-unit change in a feature, keeping other features fixed.

### 4. Explain R² score and its significance.

R² shows how much of the variation in the target is explained by the model. A value closer to 1 generally indicates a better fit.

### 5. Why is train-test splitting necessary?

It checks how well the model performs on unseen data instead of only on the data used for training.

---

## Experiment 5

### 1. What is the difference between linear regression and logistic regression?

Linear regression predicts continuous values, while logistic regression is mainly used for classification.

### 2. What is the role of the sigmoid function in logistic regression?

The sigmoid converts the model output into a value between 0 and 1, which can be treated as a probability.

### 3. Explain confusion matrix components (TP, FP, TN, FN).

- TP: predicted positive and actually positive.
- FP: predicted positive but actually negative.
- TN: predicted negative and actually negative.
- FN: predicted negative but actually positive.

### 4. Why is logistic regression used for classification instead of regression?

It models class probability using the sigmoid function and is suited to categorical outcomes.

### 5. What are common performance metrics for classification?

Accuracy, precision, recall, F1-score and confusion matrix are common metrics.

---

## Experiment 6

### 1. How does the KNN algorithm work?

KNN finds the K closest training points to a new point and assigns the majority class among them.

### 2. What is the effect of choosing a very small or very large K value?

A very small K can make the model sensitive to noise. A very large K can make the model too general and may ignore local patterns.

### 3. Which distance metrics can be used in KNN?

Euclidean, Manhattan and Minkowski distance are common examples.

### 4. Is KNN a parametric or non-parametric algorithm?

KNN is a non-parametric algorithm.

### 5. Why is feature scaling important in KNN?

Because KNN uses distance, a feature with a much larger numerical range can dominate the result if scaling is not performed.

---

## Experiment 7

### 1. What is the difference between feature selection and extraction?

Feature selection keeps a subset of the original features. Feature extraction transforms existing features into new features.

### 2. What is PCA, and why is it used?

PCA is Principal Component Analysis. It reduces dimensionality by creating principal components that preserve as much variance as possible.

### 3. How does Chi-Square feature selection work?

It measures the relationship between a categorical target and candidate features and selects features with strong dependence.

### 4. What are the advantages of dimensionality reduction?

It can reduce computation, remove irrelevant information, reduce overfitting and make visualization easier.

### 5. When would you use RFE over filter methods?

RFE can be useful when feature importance depends on a particular model and you want to select features according to that model.

---

## Experiment 8

### 1. What is the main principle behind SVM?

SVM finds a decision boundary that maximizes the margin between classes.

### 2. What are support vectors in SVM?

Support vectors are the training points closest to the decision boundary. They strongly influence the location of the boundary.

### 3. Explain the role of kernel functions in SVM.

Kernel functions allow SVM to work with non-linear relationships by representing the data in a higher-dimensional feature space.

### 4. What is the difference between linear SVM and non-linear SVM?

Linear SVM uses a straight hyperplane. Non-linear SVM uses kernels such as RBF to create more flexible boundaries.

### 5. What is the significance of parameters C and gamma in SVM?

`C` controls the penalty for misclassification. In an RBF kernel, `gamma` controls how strongly individual training points influence the decision boundary.

---

## Experiment 9

### 1. What is CART in Decision Trees?

CART stands for Classification and Regression Trees. It builds a tree by recursively splitting data.

### 2. What is the difference between Gini Index and Entropy?

Both measure impurity. Gini is used in CART and is generally faster to calculate. Entropy is based on information theory.

### 3. What is pruning in Decision Trees and why is it necessary?

Pruning removes unnecessary parts of a tree to reduce overfitting and improve generalization.

### 4. What are the advantages and limitations of Decision Trees?

They are easy to understand and can model non-linear relationships, but large trees can overfit the training data.

### 5. How do Decision Trees handle numerical and categorical data?

They split numerical features using threshold conditions. Categorical features can be handled through suitable preprocessing or tree implementations that support categories.

---

## Experiment 10

### 1. What is Deep Learning?

Deep Learning is a part of Machine Learning that uses neural networks with multiple layers to learn complex patterns.

### 2. What is an artificial neural network?

It is a model made of connected layers of artificial neurons that learn weights from training data.

### 3. Why is ReLU used in hidden layers?

ReLU introduces non-linearity and is computationally simple, which helps neural networks learn complex functions.

### 4. Why is sigmoid used in the final layer for binary classification?

Sigmoid produces a value between 0 and 1, which can represent the probability of the positive class.

### 5. What is the purpose of dropout?

Dropout randomly disables some neurons during training to reduce overfitting.

### 6. What is an epoch?

One epoch means the model has processed the complete training dataset once.

### 7. What is the purpose of validation data?

Validation data is used during training to monitor how well the model generalizes to data not directly used for weight updates.

### 8. Why is scaling used before training a neural network?

Scaling keeps input features on comparable ranges and generally makes optimization more stable.

### 9. What is a confusion matrix?

It is a table showing the counts of correct and incorrect predictions for each class.

### 10. What is the difference between training loss and validation loss?

Training loss is measured on the training data. Validation loss is measured on validation data and helps detect overfitting.

---

# Quick Dataset-Folder Reference

```text
ML_Lab/
│
├── Exp_01/
│   ├── Exp_01.ipynb
│   └── student_placement.csv
│
├── Exp_02/
│   ├── Exp_02.ipynb
│   └── training_data.csv
│
├── Exp_03/
│   ├── Exp_03.ipynb
│   └── titanic_train.csv
│
├── Exp_04/
│   ├── Exp_04.ipynb
│   └── diamonds.csv
│
├── Exp_05/
│   ├── Exp_05.ipynb
│   └── titanic_train.csv
│
├── Exp_06/
│   ├── Exp_06.ipynb
│   └── Iris.csv
│
├── Exp_07/
│   ├── Exp_07.ipynb
│   └── Iris.csv
│
├── Exp_08/
│   ├── Exp_08.ipynb
│   └── Iris.csv
│
├── Exp_09/
│   ├── Exp_09.ipynb
│   └── Iris.csv
│
└── Exp_10/
    └── Exp_10.ipynb
```

For Experiment 10, the scikit-learn breast-cancer dataset can be loaded directly, so a CSV is optional. If you want every experiment to use a local file, export that dataset to `breast_cancer.csv` once and load it with Pandas.
