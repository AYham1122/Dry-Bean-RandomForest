# Dry Bean Classification: Comparing Five Optimization Algorithms

This project compares five gradient-based optimization algorithms on the **Dry Bean Dataset**. Each optimizer trains a multi-class **Softmax Regression** model implemented manually with NumPy. The notebooks also use a Random-Forest-style ensemble strategy to make predictions more robust.

> **Scope:** This repository contains five optimizer notebooks. It does not include a separate cross-optimizer `Voting Ensemble` notebook. Hard-voting and soft-voting results reported inside each optimizer notebook combine that optimizer's own base learners.

## Dataset

The project uses the [Dry Bean Dataset on Kaggle](https://www.kaggle.com/datasets/muratkokludataset/dry-bean-dataset).

| Property | Value |
|---|---:|
| Samples | 13,611 |
| Numeric features | 16 |
| Target column | `Class` |
| Classes | 7 |
| Class names | BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA |

The dataset is imbalanced: DERMASON has 3,546 samples, while BOMBAY has 522 samples.

## Notebooks

| Notebook | Optimizer | Update approach | Epochs | Learning-rate candidates |
|---|---|---|---:|---|
| `bean-gd.ipynb` | Gradient Descent (full batch) | One update using the full training set | 500 | 3, 1, 0.3 |
| `bean-sgd.ipynb` | Stochastic Gradient Descent (SGD) | One update per training sample | 10 | 0.1, 0.03, 0.01 |
| `bean-mini-batch-gd.ipynb` | Mini-Batch Gradient Descent | One update per batch of 32 | 30 | 1.0, 0.3, 0.1 |
| `bean-rmsprop.ipynb` | RMSProp | Adaptive scaling using a moving average of squared gradients | 300 | 0.01, 0.003, 0.001 |
| `bean-adam.ipynb` | Adam | Momentum with adaptive scaling and bias correction | 300 | 0.1, 0.03, 0.01 |

The learning rate is selected using validation loss. The test set is kept for final evaluation.

## Methodology

- The task is seven-class classification using Softmax Regression.
- Gradients and optimizer updates are implemented manually with NumPy.
- Data is split using stratified sampling: 60% training, 20% validation, and 20% testing (`random_state=42`).
- Median imputation and `StandardScaler` are fitted using the training data only.
- Each optimizer notebook trains 15 base learners.
- Each base learner is trained on a bootstrap sample and a random subset of half the available features. This is a **Random-Forest-style ensemble strategy**, not a traditional decision-tree Random Forest.
- Within each optimizer notebook, the base learners' predictions are evaluated using soft voting (averaged class probabilities) and hard voting (majority class).
- There is no separate notebook that combines predictions from different optimizers.

## Evaluation

The notebooks report:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Cross-entropy Loss (for soft-voting probabilities)
- Total training time for the 15 base learners

Macro-averaged metrics are included because the seven classes are not equally represented.

## Results

Run the five notebooks and enter the measured test results below. Do not fill this table with estimates; use the values printed by each notebook.

| Optimizer | Best learning rate | Epochs | Training time (s) | Soft-voting Accuracy | Macro Precision | Macro Recall | Macro F1 | Loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GD | | 500 | | | | | | |
| SGD | | 10 | | | | | | |
| Mini-Batch GD | | 30 | | | | | | |
| RMSProp | | 300 | | | | | | |
| Adam | | 300 | | | | | | |

## Run on Kaggle

1. Open or import one of the `.ipynb` files in [Kaggle Notebooks](https://www.kaggle.com/code).
2. Click **Add Input** and attach the Dry Bean Dataset.
3. Run the notebook cells from top to bottom.
4. Review the printed metrics and plots.
5. Repeat for the other four optimizer notebooks.

The notebooks search recursively under `/kaggle/input` for a CSV or Excel dataset file, so the full dataset path does not need to be entered manually.

## Output files

Each notebook writes its results to:

```text
/kaggle/working/bean_results/
```

The output includes a JSON metrics file and an NPZ file containing validation/test probabilities and labels. Kaggle's `/kaggle/working` directory is specific to each notebook session. Download any output files you want to keep; outputs from one notebook are not automatically available in another notebook.

## Requirements

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `openpyxl` (for reading the Excel dataset)

These packages are generally available in Kaggle Notebooks.

## Limitations

- Softmax Regression is a linear classifier and may not capture nonlinear decision boundaries between similar bean classes.
- The results depend on the learning-rate candidates, epoch counts, ensemble settings, and random seed.
- The project compares five optimizers while keeping the underlying learning task and base model family consistent; it is not a comparison of five different classifier families.
