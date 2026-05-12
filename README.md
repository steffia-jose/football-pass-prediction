# Football Pass Success Prediction — Sports Analytics

## Overview
End-to-end classification pipeline using real professional football tracking data
(StatsBomb 360) to predict whether a pass will succeed or fail based on spatial
and tactical features.

## Models Compared
- Decision Tree (CART)
- Random Forest
- Naive Bayes
- K-Nearest Neighbours (KNN)

## Results

| Model | Accuracy | Sensitivity | Specificity | ROC-AUC |
|---|---|---|---|---|
| Decision Tree | 0.880 | 0.987 | 0.095 | 0.853 |
| Random Forest | 0.880 | 0.981 | 0.143 | — |
| Naive Bayes | 0.889 | 0.997 | 0.095 | — |
| KNN (k=5) | 0.875 | 0.974 | 0.143 | — |

**Best model: Random Forest (best balance of accuracy and specificity)**  
**Pruned Decision Tree: ROC-AUC = 0.853**

## Feature Engineering
12 spatial and tactical features engineered from StatsBomb 360 freeze-frame data:

- Pass length (metres)
- Pass angle (radians)
- Defenders within 5m of receiver
- Defenders within 10m of receiver
- Teammates within 5m of receiver
- Teammates within 10m of receiver
- Start and end coordinates (x, y)
- Pass height category
- Pass type category
- Match minute
- Match second

## Methodology
- Exploratory data analysis on 1,172 passes
- Addressed 88/12% class imbalance using downsampling
- Stratified 70/30 and 80/20 train-test splits
- Repeated 10-fold cross-validation for all models
- Variable importance analysis using Random Forest
- ROC-AUC evaluation for Decision Tree variants
- Feature importance grounded in football analytics literature
  (VAEP, QPass, xG models)

## Key Findings
- Pass length is the strongest predictor of pass success
- Immediate defensive pressure (defenders within 5m) more
  important than general congestion (10m zone)
- Pruned Decision Tree produces interpretable tactical decision
  rules suitable for coaching communication

## Tools & Libraries
R, caret, rpart, rpart.plot, randomForest, e1071, ggplot2

## Dataset
StatsBomb 360 — professional football freeze-frame tracking data  
1,172 passes with spatial positioning of all players on the pitch
