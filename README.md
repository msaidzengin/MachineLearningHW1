# MachineLearningHW1

This is BIL470/570 Machine Learning Homework 1, completed on 10 November 2020.

The assignment asks for logistic regression fit with mini-batch gradient descent on two UCI datasets: Connectionist Bench (Sonar, Mines vs. Rocks) and Ionosphere. Training uses the log-loss and the sigmoid function. Only NumPy and Matplotlib are used. `hw1.pdf` is the assignment, and `MuhammedSaidZengin_201111019.pdf` is the one-page report.

## Requirements

- Python 3
- NumPy
- Matplotlib

## Run

From the repository root:

```bash
python code.py
```

`code.py` trains on both datasets. For each one it shuffles the rows, keeps 80% for training, and plots test accuracy and cost against the iteration count. Plots stay open until you close the Matplotlib windows.

```bash
python example.py
```

`example.py` trains on the Ionosphere dataset only. It holds out the first 20% as test data and plots loss and test accuracy.
