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

# Create folder for all figures
if (!dir.exists("assignment_figures")) {
  dir.create("assignment_figures")
}

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
# 3. FIGURE 1: PASS LENGTH HISTOGRAM (EDA PLOT)
# -----------------------------------------------------------

p1 <- ggplot(df_mod, aes(x = pass_length_m)) +
  geom_histogram(bins = 40, fill = "#2E86AB", color = "white") +
  geom_vline(aes(xintercept = mean(pass_length_m, na.rm = TRUE)),
             linewidth = 1) +
  geom_vline(aes(xintercept = median(pass_length_m, na.rm = TRUE)),
             linewidth = 1) +
  labs(
    title = "Distribution of Pass Lengths (m)",
    x = "Pass length (m)",
    y = "Count"
  )

ggsave(
  filename = "assignment_figures/figure1_pass_length_histogram.png",
  plot     = p1,
  width    = 8,
  height   = 6,
  dpi      = 300
)

# -----------------------------------------------------------
# 4. SELECT PREDICTORS (NUMERIC ONLY, FOR KNN)
# -----------------------------------------------------------

numeric_cols <- df_mod %>%
  select(where(is.numeric)) %>%
  names()

# Remove the target from predictors
predictors <- setdiff(numeric_cols, "pass_success")

# Final modelling dataset
model_data <- df_mod %>%
  select(all_of(c("pass_success", predictors)))

# -----------------------------------------------------------
# 5. TRAIN/TEST SPLIT
# -----------------------------------------------------------

set.seed(123)
train_index <- createDataPartition(model_data$pass_success,
                                   p = 0.7,    # 70% train
                                   list = FALSE)

train_data <- model_data[train_index, ]
test_data  <- model_data[-train_index, ]

# -----------------------------------------------------------
# 6. TRAIN CONTROL (CROSS-VALIDATION)
# -----------------------------------------------------------

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
# 7. MODEL 1: DECISION TREE (rpart)
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

# Print model summary in console
dt_model

# FIGURE 2: Decision Tree tuning plot (cp vs Accuracy)
png("assignment_figures/figure2_dt_tuning.png",
    width = 2000, height = 1500, res = 300)
plot(dt_model)
dev.off()

# Test performance
dt_pred <- predict(dt_model, newdata = test_data)
dt_cm   <- confusionMatrix(dt_pred, test_data$pass_success,
                           positive = "Success")

dt_cm

# -----------------------------------------------------------
# 8. MODEL 2: RANDOM FOREST
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

# FIGURE 3: RF tuning plot (mtry vs Accuracy)
png("assignment_figures/figure3_rf_tuning.png",
    width = 2000, height = 1500, res = 300)
plot(rf_model)
dev.off()

rf_pred <- predict(rf_model, newdata = test_data)
rf_cm   <- confusionMatrix(rf_pred, test_data$pass_success,
                           positive = "Success")

rf_cm

# Variable importance for report (best model)
rf_imp <- varImp(rf_model)

# FIGURE 5: Random Forest variable importance
png("assignment_figures/figure5_rf_variable_importance.png",
    width = 2000, height = 1500, res = 300)
plot(rf_imp, main = "Random Forest Variable Importance")
dev.off()

# -----------------------------------------------------------
# 9. MODEL 3: NAIVE BAYES
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
nb_cm   <- confusionMatrix(nb_pred, test_data$pass_success,
                           positive = "Success")

nb_cm

# (No special figure here – the main thing is performance + table.)

# -----------------------------------------------------------
# 10. MODEL 4: K-NEAREST NEIGHBOURS (KNN)
# -----------------------------------------------------------

set.seed(123)
knn_grid <- expand.grid(
  k = seq(3, 25, by = 2)
)

knn_model <- train(
  form,
  data       = train_data,
  method     = "knn",
  trControl  = ctrl,
  preProcess = c("center", "scale"),  # KNN needs scaling
  tuneGrid   = knn_grid,
  metric     = "Accuracy"
)

knn_model

# FIGURE 4: KNN tuning plot (k vs Accuracy)
png("assignment_figures/figure4_knn_tuning.png",
    width = 2000, height = 1500, res = 300)
plot(knn_model)
dev.off()

knn_pred <- predict(knn_model, newdata = test_data)
knn_cm   <- confusionMatrix(knn_pred, test_data$pass_success,
                            positive = "Success")

knn_cm

# -----------------------------------------------------------
# 11. COLLECT ACCURACY FROM ALL MODELS (FOR TABLE + FIGURE)
# -----------------------------------------------------------

model_summary <- tibble(
  Model       = c("Decision Tree", "Random Forest", "Naive Bayes", "KNN"),
  Accuracy    = c(as.numeric(dt_cm$overall["Accuracy"]),
                  as.numeric(rf_cm$overall["Accuracy"]),
                  as.numeric(nb_cm$overall["Accuracy"]),
                  as.numeric(knn_cm$overall["Accuracy"])),
  Sensitivity = c(as.numeric(dt_cm$byClass["Sensitivity"]),
                  as.numeric(rf_cm$byClass["Sensitivity"]),
                  as.numeric(nb_cm$byClass["Sensitivity"]),
                  as.numeric(knn_cm$byClass["Sensitivity"])),
  Specificity = c(as.numeric(dt_cm$byClass["Specificity"]),
                  as.numeric(rf_cm$byClass["Specificity"]),
                  as.numeric(nb_cm$byClass["Specificity"]),
                  as.numeric(knn_cm$byClass["Specificity"]))
)

model_summary

# -----------------------------------------------------------
# 12. FIGURE 6: MODEL ACCURACY COMPARISON BARPLOT
# -----------------------------------------------------------

p_models <- ggplot(model_summary,
                   aes(x = Model, y = Accuracy)) +
  geom_col() +
  ylim(0, 1) +
  labs(
    title = "Model Accuracy on Test Set",
    x = "Model",
    y = "Accuracy"
  )

ggsave(
  filename = "assignment_figures/figure6_model_accuracy_comparison.png",
  plot     = p_models,
  width    = 8,
  height   = 6,
  dpi      = 300
)

# (Optional) You can also write the model_summary table to CSV for easy copy into Word:
# write_csv(model_summary, "assignment_figures/table_model_summary.csv")
