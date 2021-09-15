# Supervised Learning Algorithms

*Supervised learning* is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. 
A supervised learning algorithm analyzes the training data and produces an inferred function, which can be used for mapping new examples.

The most popular supervised learning tasks are: *Regression* and *Classification*.
- The result of solving the *regression* task is a model that can make *numerical predictions*. For example:
  - Real estate value prediction based on house location.
  - Predicting your company's revenue next year
- The result of solving the *classification* task is a model that can make *classes predictions*. For example:
  - Spam detection.
  - Classifying news articles.
  - Analyzing images of products on a production line to automatically classify them.
  - Predicting the probability of cancer based on blood tests.
- The line between these tasks is often fuzzy.

Classification algorithms can be also devided to **hard** and **soft**:
- **Hard classification algorithms** predict whether a given data point is part of a particular class **without producing the probability estimation**.
- **Soft classification algorithms** in turn, also estimate the class conditional **probabilities**.

Classification algorithms can be also devided by the number of classes to classify:
- **Binary classification** - only two classes.
- **Multiclass classification** - more than two classes.
  - **Multilabel classification** (Multilabel-multiclass) - multiple classes, but classes are binary (people on image). Result - [0, 0, 1] or [1, 0, 1].
  - **Multioutput classification** (Multioutput-multiclass) also known as **multitask classification** - multiple classes, but classes are not binary. Result - [5, 0, -1] or [7, 0, 0].

Some algorithms are designed only for binary classification problems (for example, *Logistic Regression* or *SVM*). So, they cannot be used for multi-class classification tasks directly. 
Instead, heuristic methods can be used to split a multi-class classification problem into multiple binary classification datasets and train a binary classification model each:
- **OvR** (one-vs-rest) - sometimes **OvA** (one-vs-all) - you have to train N classifiers for N classes, but on full dataset.
- **OvO** (one-vs-one) - you have to train N*(N-1)/2 classifiers for N classes, but on subsamples from your dataset. Better suited for unbalanced samples.

Next, the following algorithms will be reviewed or mentioned (note, that *all of them solve both classification and regression task*, except of *Linear Regression (only Regression)* and *Logistic Regression (only Classification*))*:
- *Linear Regression*
- *Logistic Regression*
- *SVM* - Support Vector Machines
- *kNN* - k-Nearest Neighbors
- *Decision Tree*
- Ensemble methods:
  - *Bagging* and *Pasting*
  - *Random Forest* and *Extra Trees*
  - *Boosting*
  - *Staking* and *Blending*



# Ensemble Methods

Ensemble methods are techniques that create multiple models and then combine them to produce improved results. Ensemble methods usually produces more accurate solutions than a single model would.