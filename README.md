# Predicting air pressure system failures in Scania trucks

Which trucks should go to the workshop? A missed failure costs 500; an unnecessary inspection costs 10. I built a model that decides, and set its decision rule on that cost rather than on accuracy.

## Result

I locked 16,000 trucks (375 of them failures) away at the start and used them only once, at the very end, to check the final model on trucks it had never seen:

| Approach | Cost |
|---|---|
| Flag no trucks | 187,500 |
| Flag every truck | 156,250 |
| **Gradient boosting, cost-tuned threshold (this project)** | **11,040** |
| 2016 IDA Industrial Challenge: 1st / 2nd / 3rd | 9,920 / 10,900 / 11,480 |

The model caught **364 of 375 failing trucks (97%)**, with 554 false alarms among 15,625 trucks with healthy APS.

## The problem

The data comes from heavy Scania trucks in everyday use. The positive class is failures of one component of the air pressure system (APS), which supplies air for braking and gear changes. Failures are rare: 1.7% of trucks. A model that predicts "healthy" APS for every truck is 98.3% accurate and useless, so I evaluated everything on Scania's cost metric: `cost = 10 × false alarms + 500 × missed failures`. The comparison group isn’t healthy trucks: it’s trucks whose failures were in other components.

## Data

[APS Failure at Scania Trucks](https://doi.org/10.24432/C51S51), UCI Machine Learning Repository (2016). 60,000 training and 16,000 test trucks; 170 anonymised sensor columns. UCI lists the dataset under CC BY 4.0; the files carry Scania's GPL-3.0 notice, kept unchanged in `data/raw/`.

## Method

Every choice (which model, which settings, which threshold) was made using the 60,000 training trucks only. To test models fairly within that data, I split it into five parts ("folds"), trained on four and tested on the fifth, and rotated until every truck had been tested once. Each fold kept the same share of failures (1.7%), so the five tests were comparable. The 16,000 locked-away trucks were used once, at the end, with the model and threshold already fixed.

1. **Baseline.** A dummy model that always predicts trucks with "healthy" APS costs 500,000 on the training data.
2. **Logistic regression** inside a scikit-learn Pipeline (median imputation, scaling, balanced class weights), so the preprocessing is learned from the training folds only, never from the fold being tested: cost about 60,000.
3. **Gradient boosting** (`HistGradientBoostingClassifier`, which handles missing values itself, so no filling-in is needed): cost about 42,000.
4. **Threshold.** I chose the decision threshold that minimised cost: 0.11 instead of the default 0.5. For gradient boosting this cut the cost by about a third.
5. **Stability.** Re-running the five-fold test with the trucks shuffled differently gave 41,500–45,500.

## Findings

- **The decision rule matters as much as the model.** At the default threshold, gradient boosting was no better than logistic regression.
- **Missing values are informative, but not extra information.** Several sensors are missing for about 80% of trucks with healthy APS but under 10% of failed ones. Adding "was missing" flags to logistic regression changed the cost by less than the variation between shuffles: the other sensors already carry the signal.
- **Cross-validation predicted the test result:** cost per truck of 0.70 in cross-validation and 0.69 on test.

## Limitations

- **The sensors are anonymised,** so the model can't be interpreted physically.
- **Why missingness differs between classes is unknown.** If it reflects how failed trucks' records were collected rather than how the trucks behave, real-world performance would be lower than the test score.
- **Default settings.** I didn't tune gradient boosting; the threshold is the only tuned choice.
- **The 2016 comparison is indicative.** Like the challenge entries, I scored the test set once, with everything fixed beforehand, but with a 2026 library and with the challenge results already published.
- **The data is an expert-selected subset,** and the comparison group is trucks with other failures, not a random sample of a fleet. Real-world precision would likely be lower than on this test set.

## Repository

- `notebooks/01_explore.ipynb`: class balance and missing values
- `notebooks/02_baseline.ipynb`: cost function, baselines, logistic regression, gradient boosting, threshold and stability
- `notebooks/03_final_test.ipynb`: the single test-set evaluation

To run: `pip install -r requirements.txt`, then the notebooks in order.