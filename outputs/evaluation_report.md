# Evaluation Report — Random Forest

## Results Summary

| Metric | Value |
|--------|-------|
| Best Model | Random Forest |
| Test Accuracy | 0.8733 |
| CV Score (5-fold) | 0.8737 |
| Random Forest Accuracy | 0.8733 |
| Logistic Regression Accuracy | 0.8656 |

## Classification Report

```
              precision    recall  f1-score   support

           0       0.84      0.93      0.88     20711
           1       0.92      0.82      0.87     20666

    accuracy                           0.87     41377
   macro avg       0.88      0.87      0.87     41377
weighted avg       0.88      0.87      0.87     41377

```

## Confusion Matrix

|  | Predicted 0 | Predicted 1 |
|--|-------------|-------------|
| Actual 0 | 19227 | 1484 |
| Actual 1 | 3757 | 16909 |

## Top 10 Feature Importances

| Feature | Importance |
|---------|-----------|
| num_reviews | 0.4129 |
| num_comments | 0.2491 |
| avg_rating | 0.0700 |
| price | 0.0633 |
| is_paid | 0.0582 |
| course_age_years | 0.0499 |
| publish_year | 0.0374 |
| language | 0.0293 |
| num_lectures | 0.0125 |
| content_length_min | 0.0088 |
