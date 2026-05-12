
# ============================================================

# ---------- 0) Packages ----------
pkgs <- c("readr","dplyr","ggplot2","tidyr","caret","rpart",
          "rpart.plot","MLmetrics","GGally","pROC")
to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if(length(to_install)) install.packages(to_install, dependencies = TRUE)
invisible(lapply(pkgs, library, character.only = TRUE))

set.seed(42)
outdir <- "assignment2_outputs"
if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)

# ---------- 1) Locate CSV robustly ----------
# We try a few common locations so you don't have to worry about setwd()
candidate_paths <- c(
  "assignment_outputs/engineered_pass_features_clean.csv",  # parent dir
  "engineered_pass_features_clean.csv"                      # current dir is assignment_outputs
)
csv_path <- candidate_paths[file.exists(candidate_paths)][1]
if (is.na(csv_path)) stop("Could not find engineered_pass_features_clean.csv in expected locations.")

message("Using CSV at: ", normalizePath(csv_path, winslash = "/"))

# ---------- 2) Load & Select Modelling Features ----------
df <- readr::read_csv(csv_path, show_col_types = FALSE)

# Keep engineered numeric + relevant categorical; drop ID/team/player to avoid leakage
df_model <- df %>%
  transmute(
    pass_success = factor(pass_success, levels = c(0,1), labels = c("Fail","Success")),
    pass_length_m, pass_angle_rad,
    defenders_near_5m, defenders_near_10m,
    teammates_near_5m, teammates_near_10m,
    pass_height = factor(pass_height),
    pass_type   = factor(pass_type)
  )

# Save class distribution
class_dist <- df_model %>% count(pass_success) %>%
  mutate(prop = n/sum(n))
readr::write_csv(class_dist, file.path(outdir, "class_distribution.csv"))

# ---------- 3) EDA (modelling-focused) ----------
# Histograms by class
p_len <- ggplot(df_model, aes(pass_length_m, fill = pass_success)) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 30) +
  labs(title="Pass length by outcome", x="Pass length (m)", y="Count") +
  theme_minimal(base_size=12)
ggsave(file.path(outdir, "eda_len_by_class.png"), p_len, width=6, height=4, dpi=300)

p_ang <- ggplot(df_model, aes(pass_angle_rad, fill = pass_success)) +
  geom_histogram(position = "identity", alpha = 0.5, bins = 30) +
  labs(title="Pass angle by outcome", x="Pass angle (rad)", y="Count") +
  theme_minimal(base_size=12)
ggsave(file.path(outdir, "eda_angle_by_class.png"), p_ang, width=6, height=4, dpi=300)

# Boxplots for contextual features by class
long_press <- df_model %>%
  pivot_longer(c(defenders_near_5m, defenders_near_10m,
                 teammates_near_5m, teammates_near_10m),
               names_to = "feature", values_to = "value")
p_box <- ggplot(long_press, aes(pass_success, value)) +
  geom_boxplot() +
  facet_wrap(~ feature, scales = "free_y") +
  labs(title="Context features by outcome", x=NULL, y=NULL) +
  theme_minimal(base_size=12)
ggsave(file.path(outdir, "eda_context_box_by_class.png"), p_box, width=8, height=5, dpi=300)

# Categorical distributions (stacked proportions)
p_height <- ggplot(df_model, aes(pass_height, fill = pass_success)) +
  geom_bar(position="fill") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(title="Pass height: class proportions", x=NULL, y="Proportion") +
  theme_minimal(base_size=12)
ggsave(file.path(outdir, "eda_height_prop.png"), p_height, width=6, height=4, dpi=300)

p_type <- ggplot(df_model, aes(pass_type, fill = pass_success)) +
  geom_bar(position="fill") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(title="Pass type: class proportions", x=NULL, y="Proportion") +
  theme_minimal(base_size=12) +
  theme(axis.text.x = element_text(angle=25, hjust=1))
ggsave(file.path(outdir, "eda_type_prop.png"), p_type, width=8, height=4.5, dpi=300)

# Correlation among numeric inputs
num_vars <- df_model %>% select(pass_length_m, pass_angle_rad,
                                defenders_near_5m, defenders_near_10m,
                                teammates_near_5m, teammates_near_10m)
p_corr <- GGally::ggcorr(num_vars, label=TRUE, hjust=0.8, size=3)
ggsave(file.path(outdir, "eda_numeric_corr.png"), p_corr, width=6.5, height=6, dpi=300)

# ---------- 4) Train/Test Split (stratified) ----------
set.seed(42)
idx <- caret::createDataPartition(df_model$pass_success, p = 0.8, list = FALSE)
train <- df_model[idx, ]
test  <- df_model[-idx, ]

# Handle class imbalance in training via downsampling
train_bal <- caret::downSample(x = train %>% select(-pass_success),
                               y = train$pass_success,
                               yname = "pass_success")

# ---------- 5) Decision Tree with CV (tune cp) ----------
ctrl <- caret::trainControl(method = "cv", number = 5,
                            classProbs = TRUE,
                            summaryFunction = twoClassSummary, # uses ROC
                            savePredictions = "final")

set.seed(42)
fit <- caret::train(
  pass_success ~ .,
  data = train_bal,
  method = "rpart",
  trControl = ctrl,
  metric = "ROC",
  tuneLength = 15
)

sink(file.path(outdir, "model_cv_results.txt")); print(fit); sink()

# Tree plot
png(file.path(outdir, "tree_plot.png"), width=1000, height=700)
rpart.plot::rpart.plot(fit$finalModel, type = 3, extra = 104, fallen.leaves = TRUE)
dev.off()

# Variable importance
imp <- varImp(fit)
gg_imp <- ggplot(imp) + ggtitle("Variable importance (caret/rpart)")
ggsave(file.path(outdir, "variable_importance.png"), gg_imp, width=6, height=4.5, dpi=300)
saveRDS(fit, file.path(outdir, "decision_tree_caret_fit.rds"))

# ---------- 6) Evaluate on untouched test set ----------
pred_prob <- predict(fit, newdata = test, type = "prob")[,"Success"]
pred_cls  <- factor(ifelse(pred_prob >= 0.5, "Success", "Fail"),
                    levels = c("Fail","Success"))

cm <- caret::confusionMatrix(pred_cls, test$pass_success, positive = "Success")
sink(file.path(outdir, "test_confusion_matrix.txt")); print(cm); sink()

# Extra metrics
acc  <- MLmetrics::Accuracy(pred_cls, test$pass_success)
prec <- MLmetrics::Precision(pred_cls, test$pass_success, positive = "Success")
rec  <- MLmetrics::Recall(pred_cls, test$pass_success, positive = "Success")
f1   <- MLmetrics::F1_Score(pred_cls, test$pass_success, positive = "Success")

# ROC-AUC on test
roc_obj <- pROC::roc(response = test$pass_success, predictor = pred_prob, levels = c("Fail","Success"))
auc_val <- as.numeric(pROC::auc(roc_obj))

metrics <- data.frame(Accuracy=acc, Precision=prec, Recall=rec, F1=f1, AUC=auc_val)
readr::write_csv(metrics, file.path(outdir, "test_metrics.csv"))
png(file.path(outdir, "test_roc_curve.png"), width=650, height=500)
plot(roc_obj, main = sprintf("Test ROC (AUC = %.3f)", auc_val))
dev.off()

# Save predictions (for appendix / reproducibility)
pred_df <- test %>%
  mutate(pred_prob_success = pred_prob,
         pred_class = pred_cls)
readr::write_csv(pred_df, file.path(outdir, "test_predictions.csv"))

# ---------- 7) Session info (reproducibility) ----------
sink(file.path(outdir, "sessionInfo.txt")); print(sessionInfo()); sink()

message("All done. Outputs saved to: ", normalizePath(outdir, winslash = "/"))
