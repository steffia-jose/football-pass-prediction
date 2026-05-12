# -----------------------------------------------------------
# 0. PACKAGES
# -----------------------------------------------------------

# install.packages("tidyverse")
# install.packages("caret")
# install.packages("rpart")
# install.packages("randomForest")
# install.packages("naivebayes")
# install.packages("pROC")

library(tidyverse)
library(caret)
library(rpart)
library(randomForest)
library(naivebayes)
library(pROC)

set.seed(123)

# -----------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------

df <- read_csv("engineered_pass_features_clean.csv",
               show_col_types = FALSE)

# Quick check
glimpse(df)

# -----------------------------------------------------------
# 2. DEFINE TARGET (BINARY) + BASIC PREP
# -----------------------------------------------------------
# We use pass_success (0/1) as the target.
# 0 = Fail, 1 = Success

df_mod <- df %>%
  drop_na(pass_success) %>%              # just in case
  mutate(
    pass_success = factor(
      pass_success,
      levels = c(0, 1),
      labels = c("Fail", "Success")
    )
  )

# -----------------------------------------------------------
# 3. SELECT PREDICTORS
# -----------------------------------------------------------
# To keep life simple (especially for KNN), we will use ONLY
# NUMERIC FEATURES as predictors.
# (Trees & RF can use factors too, but KNN cannot.)
# This is a reasonable, defensible choice for the assignment.

numeric_cols <- df_mod %>%
  select(where(is.numeric)) %>%
  names()

# Remove the target from predictors
predictors <- setdiff(numeric_cols, "pass_success")

# Final modelling dataset
model_data <- df_mod %>%
  select(all_of(c("pass_success", predictors)))

# -----------------------------------------------------------
# 4. TRAIN/TEST SPLIT
# -----------------------------------------------------------

set.seed(123)
train_index <- createDataPartition(model_data$pass_success,
                                   p = 0.7,    # 70% train
                                   list = FALSE)

train_data <- model_data[train_index, ]
test_data  <- model_data[-train_index, ]

# -----------------------------------------------------------
# 5. TRAIN CONTROL (CROSS-VALIDATION)
# -----------------------------------------------------------
# Keep it simple: use Accuracy as the main metric

ctrl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5
)

# Build formula: pass_success ~ all numeric predictors
form <- as.formula(
  paste("pass_success ~", paste(predictors, collapse = " + "))
)

# -----------------------------------------------------------
# 6. MODEL 1: DECISION TREE (rpart)
# -----------------------------------------------------------

set.seed(123)
dt_grid <- expand.grid(
  cp = seq(0.001, 0.05, by = 0.005)
)

dt_model <- train(
  form,
  data      = train_data,
  method    = "rpart",
  trControl = ctrl,
  tuneGrid  = dt_grid,
  metric    = "Accuracy"
)

dt_model
plot(dt_model)  # cp vs Accuracy

# Test performance
dt_pred <- predict(dt_model, newdata = test_data)
confusionMatrix(dt_pred, test_data$pass_success, positive = "Success")

# -----------------------------------------------------------
# 7. MODEL 2: RANDOM FOREST
# -----------------------------------------------------------

set.seed(123)
rf_grid <- expand.grid(
  mtry = c(2, 4, 6, 8)   # number of variables tried at each split
)

rf_model <- train(
  form,
  data      = train_data,
  method    = "rf",
  trControl = ctrl,
  tuneGrid  = rf_grid,
  metric    = "Accuracy",
  ntree     = 300
)

rf_model
plot(rf_model)

rf_pred <- predict(rf_model, newdata = test_data)
confusionMatrix(rf_pred, test_data$pass_success, positive = "Success")

# Variable importance for report (best model)
rf_imp <- varImp(rf_model)
plot(rf_imp)

# -----------------------------------------------------------
# 8. MODEL 3: NAIVE BAYES
# -----------------------------------------------------------

set.seed(123)

nb_model <- train(
  form,
  data      = train_data,
  method    = "naive_bayes",
  trControl = ctrl,
  metric    = "Accuracy"
)

nb_model

nb_pred <- predict(nb_model, newdata = test_data)
confusionMatrix(nb_pred, test_data$pass_success, positive = "Success")

# -----------------------------------------------------------
# 9. MODEL 4: K-NEAREST NEIGHBOURS (KNN)
# -----------------------------------------------------------

set.seed(123)
knn_grid <- expand.grid(
  k = seq(3, 25, by = 2)
)

knn_model <- train(
  form,
  data      = train_data,
  method    = "knn",
  trControl = ctrl,
  preProcess = c("center", "scale"),  # KNN needs scaling
  tuneGrid  = knn_grid,
  metric    = "Accuracy"
)

knn_model
plot(knn_model)

knn_pred <- predict(knn_model, newdata = test_data)
confusionMatrix(knn_pred, test_data$pass_success, positive = "Success")

# -----------------------------------------------------------
# 10. COLLECT ACCURACY FROM ALL MODELS (FOR COMPARISON TABLE)
# -----------------------------------------------------------

dt_cm  <- confusionMatrix(dt_pred,  test_data$pass_success, positive = "Success")
rf_cm  <- confusionMatrix(rf_pred,  test_data$pass_success, positive = "Success")
nb_cm  <- confusionMatrix(nb_pred,  test_data$pass_success, positive = "Success")
knn_cm <- confusionMatrix(knn_pred, test_data$pass_success, positive = "Success")

model_summary <- tibble(
  Model       = c("Decision Tree", "Random Forest", "Naive Bayes", "KNN"),
  Accuracy    = c(dt_cm$overall["Accuracy"],
                  rf_cm$overall["Accuracy"],
                  nb_cm$overall["Accuracy"],
                  knn_cm$overall["Accuracy"]),
  Sensitivity = c(dt_cm$byClass["Sensitivity"],
                  rf_cm$byClass["Sensitivity"],
                  nb_cm$byClass["Sensitivity"],
                  knn_cm$byClass["Sensitivity"]),
  Specificity = c(dt_cm$byClass["Specificity"],
                  rf_cm$byClass["Specificity"],
                  nb_cm$byClass["Specificity"],
                  knn_cm$byClass["Specificity"])
)

model_summary
