library(tidyverse)
library(xgboost)
library(caret)
library(SHAPforxgboost)
# 1. データの読み込み
humor_raw <- read_csv("Humor.csv")
cmu_raw   <- read_csv("Sublex_CMU.csv")

# 2. Humorデータのクリーニング
humor_clean <- humor_raw %>%
  select(Word, Humor_Overall_Enge) %>%
  mutate(
    word_lower = str_to_lower(str_trim(Word)),
    humor_score = as.numeric(Humor_Overall_Enge)
  ) %>%
  filter(!is.na(humor_score), !is.na(word_lower)) %>%
  distinct(word_lower, .keep_all = TRUE) %>%
  select(word_lower, humor_score)

# 3. Sublex_CMUデータのクリーニング
phoneme_cols <- setdiff(names(cmu_raw), c("word", "frequency"))

cmu_clean <- cmu_raw %>%
  mutate(
    word_lower = str_to_lower(str_trim(word)),
    across(all_of(phoneme_cols), ~ replace_na(as.numeric(.), 0))
  ) %>%
  distinct(word_lower, .keep_all = TRUE)

# 4. データの結合
merged_df <- inner_join(humor_clean, cmu_clean, by = "word_lower")

# 5. 音素の出現回数を単語長（総音素数）で正規化
analysis_df <- merged_df %>%
  mutate(total_phonemes = rowSums(across(all_of(phoneme_cols)))) %>%
  filter(total_phonemes > 0) %>%
  mutate(across(all_of(phoneme_cols), ~ . / total_phonemes))

# 6. 結合結果の確認
cat("結合後の単語数 (N):", nrow(analysis_df), "\n")
summary(analysis_df$humor_score)

set.seed(42)

# 1. 全データモデル（中央値分割・5-fold CV）
median_score <- median(analysis_df$humor_score)

df_full <- analysis_df %>%
  mutate(Humor_Class = factor(ifelse(humor_score >= median_score, "High", "Low"),
                              levels = c("High", "Low"))) %>%
  select(Humor_Class, all_of(phoneme_cols))

ctrl <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = "final",
  classProbs = TRUE
)

xgb_grid <- expand.grid(
  nrounds = c(100, 200),
  max_depth = c(3, 6),
  eta = c(0.05, 0.1),
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8
)

model_full <- train(
  Humor_Class ~ .,
  data = df_full,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = xgb_grid,
  verbosity = 0
)

best_full <- model_full$results %>% arrange(desc(Accuracy)) %>% slice(1)
cat("=== Full Dataset Model (N =", nrow(df_full), ") ===\n")
cat(sprintf("Accuracy: %.1f%% (SD = %.1f%%)\n", best_full$Accuracy * 100, best_full$AccuracySD * 100))
print(confusionMatrix(model_full$pred$pred, model_full$pred$obs))


# 2. 上位10%・下位10%モデル（Extreme 10%・5-fold CV）
q10 <- quantile(analysis_df$humor_score, 0.10)
q90 <- quantile(analysis_df$humor_score, 0.90)

df_extreme <- analysis_df %>%
  filter(humor_score <= q10 | humor_score >= q90) %>%
  mutate(Humor_Class = factor(ifelse(humor_score >= q90, "High", "Low"),
                              levels = c("High", "Low"))) %>%
  select(Humor_Class, all_of(phoneme_cols))

model_extreme <- train(
  Humor_Class ~ .,
  data = df_extreme,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = xgb_grid,
  verbosity = 0
)

best_ext <- model_extreme$results %>% arrange(desc(Accuracy)) %>% slice(1)
cat("\n=== Top & Bottom 10% Model (N =", nrow(df_extreme), ") ===\n")
cat(sprintf("Accuracy: %.1f%% (SD = %.1f%%)\n", best_ext$Accuracy * 100, best_ext$AccuracySD * 100))
print(confusionMatrix(model_extreme$pred$pred, model_extreme$pred$obs))


# 3. 特徴量重要度（caret）と方向性の算出（上位・下位10%モデル)

imp_caret <- varImp(model_extreme, scale = TRUE)$importance %>%
  rownames_to_column("Phoneme") %>%
  rename(Importance = Overall) %>%
  arrange(desc(Importance))

dir_df <- df_extreme %>%
  group_by(Humor_Class) %>%
  summarise(across(all_of(phoneme_cols), mean)) %>%
  pivot_longer(-Humor_Class, names_to = "Phoneme", values_to = "Mean_Freq") %>%
  pivot_wider(names_from = Humor_Class, values_from = Mean_Freq) %>%
  mutate(Directionality = ifelse(High > Low, "High (Funny)", "Low (Not Funny)"))

feature_table <- left_join(imp_caret, dir_df, by = "Phoneme") %>%
  select(Phoneme, Importance, Directionality, High, Low)

cat("\n=== Top 10 Phonemes (Extreme 10% Model) ===\n")
print(head(feature_table, 10))

# 4. ポスター用 SHAP Summary Plot の作成
X_mat <- as.matrix(df_extreme %>% select(all_of(phoneme_cols)))
y_vec <- ifelse(df_extreme$Humor_Class == "High", 1, 0)

dtrain <- xgb.DMatrix(data = X_mat, label = y_vec)

best_p <- model_extreme$bestTune

params <- list(
  objective = "binary:logistic",
  max_depth = best_p$max_depth,
  eta = best_p$eta,
  gamma = best_p$gamma,
  colsample_bytree = best_p$colsample_bytree,
  min_child_weight = best_p$min_child_weight,
  subsample = best_p$subsample
)

set.seed(42)
xgb_for_shap <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_p$nrounds,
  verbose = 0
)

shap_values <- shap.values(xgb_model = xgb_for_shap, X_train = X_mat)
shap_long <- shap.prep(xgb_model = xgb_for_shap, X_train = X_mat, top_n = 7)

p_shap <- shap.plot.summary(shap_long) +
  labs(
    title = "SHAP Summary Plot: Top Phonemes Predicting Word Humor",
    subtitle = "Positive SHAP value = Higher Humor Rating (Top 10% vs. Bottom 10%)",
    x = "",
    y = "SHAP value (impact on model output)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "bottom",
    legend.key.width = unit(2.5, "cm"),
    legend.key.height = unit(0.4, "cm")
  )

ggsave("shap_summary_plot.png", plot = p_shap, width = 9, height = 6.5, dpi = 300)
