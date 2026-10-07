# Contemporary Artificial Intelligence: Experiment 1

Text classification experiment by 王亦昕 (10245000462).

The experiment compares two word-level TF-IDF representations with three classical classifiers:

- Multinomial Naive Bayes, `alpha` in `{0.1, 0.5, 1.0}`
- Logistic Regression, `C` in `{0.5, 1.0, 2.0}`
- Linear SVM, `C` in `{0.5, 1.0, 2.0}`

The random seed is fixed at 42. The labeled data is split once using an 80/20 stratified split. Model selection uses validation macro-F1; the unlabeled test set is not used during tuning.

## Data

Place the two course-provided files next to `experiment.py`:

```text
train_data.csv
test_data_unlabeled.csv
```

Expected schemas:

- `train_data.csv`: `text,target`
- `test_data_unlabeled.csv`: `text`

The course data is not redistributed in this repository.

## Run

```bash
python -m pip install -r requirements.txt
python experiment.py
```

All generated artifacts are written to `outputs/`, including:

- `predictions.csv`: 2,457 predictions without a header
- `validation_results.csv`: all 18 validation results
- `summary.json`: selected configuration and environment
- diagnostic plots, classification report, confusion matrix, and error examples
- `best_model.joblib`: the final pipeline retrained on all labeled data

## Result

The selected configuration was word unigram+bigram TF-IDF with Linear SVM (`C=1.0`). It achieved validation accuracy `0.9417` and macro-F1 `0.9418` on the fixed stratified split.
