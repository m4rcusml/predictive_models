import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

print("\n========== LEITURA E FEATURE ENGINEERING (RESEARCH+KAGGLE CONTEXTO) ==========")
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

for df in [train, test]:
    # Features originais e "clássicas"
    df['funding_per_round'] = df['funding_total_usd'] / (df['funding_rounds'] + 1e-9)
    df['milestones_per_year'] = df['milestones'] / (df['age_last_milestone_year'].fillna(df['age_last_milestone_year'].median()) + 1)
    df['funding_per_year'] = df['funding_total_usd'] / (df['age_last_funding_year'].fillna(df['age_last_funding_year'].median()) + 1)
    df['participants_per_round'] = df['avg_participants'] / (df['funding_rounds'] + 1e-9)
    df['relationships_per_round'] = df['relationships'] / (df['funding_rounds'] + 1e-9)
    df['is_CA_software'] = df['is_CA'] * df['is_software']
    df['is_NY_advertising'] = df['is_NY'] * df['is_advertising']
    df['is_MA_biotech'] = df['is_MA'] * df['is_biotech']
    df['is_high_funding'] = (df['funding_total_usd'] > df['funding_total_usd'].median()).astype(int)
    df['is_high_milestones'] = (df['milestones'] > df['milestones'].median()).astype(int)
    df['time_to_first_milestone_after_funding'] = (
        df['age_first_milestone_year'] - df['age_first_funding_year']
    )
    df['milestone_per_million'] = df['milestones'] / (df['funding_total_usd'] / 1e6 + 1)
    
    # --- Features avançadas de pesquisa em sucesso de startups ----
    df['has_multiple_funding_types'] = ((df['has_VC'] + df['has_angel']) >= 2).astype(int)
    df['advanced_funding_rounds'] = ((df['has_roundC'] + df['has_roundD']) >= 1).astype(int)
    df['funding_progression_score'] = (df['has_roundA'] + df['has_roundB'] + df['has_roundC'] + df['has_roundD'])
    df['funding_velocity'] = df['funding_total_usd'] / (df['age_last_funding_year'].fillna(1) + 1)
    df['milestone_momentum'] = df['milestones'] / (df['age_last_milestone_year'].fillna(1) + 1)
    df['network_density'] = df['relationships'] * df['avg_participants'] / (df['funding_rounds'] + 1)
    df['ecosystem_strength'] = df['relationships'] + df['milestones'] + df['funding_rounds']
    df['traction_score'] = (df['funding_total_usd'] / 1000000) * df['milestones'] * df['relationships']
    df['is_high_traction'] = (df['traction_score'] > df['traction_score'].quantile(0.75)).astype(int)
    df['efficiency_ratio'] = df['milestones'] / (df['funding_total_usd'] / 1000000 + 1)
    df['capital_efficiency'] = df['relationships'] / (df['funding_total_usd'] / 1000000 + 1)
    df['funding_milestone_alignment'] = np.abs(
        df['age_last_funding_year'].fillna(0) - df['age_last_milestone_year'].fillna(0)
    )
    df['is_well_timed_funding'] = (df['funding_milestone_alignment'] <= 1).astype(int)
    df['is_high_growth_sector'] = (
        df['is_software'] + df['is_biotech'] + df['is_mobile'] + df['is_enterprise']
    ).astype(int)
    df['is_traditional_sector'] = (
        df['is_consulting'] + df['is_advertising']
    ).astype(int)
    df['is_innovation_hub'] = (df['is_CA'] + df['is_NY'] + df['is_MA']).astype(int)
    df['innovation_hub_tech'] = df['is_innovation_hub'] * df['is_high_growth_sector']
    df['scale_readiness'] = (
        (df['funding_rounds'] >= 3).astype(int) + 
        (df['relationships'] >= df['relationships'].median()).astype(int) +
        (df['milestones'] >= df['milestones'].median()).astype(int)
    )
    df['diversified_funding'] = df['funding_progression_score'] * df['has_multiple_funding_types']
    df['sustainable_growth_pattern'] = (
        df['is_well_timed_funding'] * df['efficiency_ratio'] * df['network_density']
    )
    df['competitive_moat'] = df['milestones'] * df['relationships'] / (df['funding_rounds'] + 1)
    df['market_position_strength'] = (
        df['funding_total_usd'] / (df['funding_total_usd'].median() + 1)
    ) * df['milestone_momentum']

# Atualização das listas de features
cont_features = [
    'age_first_funding_year', 'age_last_funding_year',
    'age_first_milestone_year', 'age_last_milestone_year',
    'relationships', 'funding_rounds', 'funding_total_usd',
    'milestones', 'avg_participants', 'relationships_per_round',
    'participants_per_round', 'funding_per_year', 'milestones_per_year',
    'funding_per_round', 'time_to_first_milestone_after_funding',
    # Avançadas:
    'funding_velocity', 'milestone_momentum', 'network_density', 'ecosystem_strength',
    'traction_score', 'efficiency_ratio', 'capital_efficiency', 'funding_milestone_alignment',
    'competitive_moat', 'market_position_strength', 
    'funding_progression_score', 'scale_readiness'
]
cat_features = ['category_code']
bin_features = [col for col in train.columns if col.startswith('is_') or col.startswith('has_') and col != 'is_othercategory']
bin_features += [
    'has_multiple_funding_types', 'advanced_funding_rounds', 'is_high_traction',
    'is_well_timed_funding', 'is_high_growth_sector', 'is_traditional_sector',
    'is_innovation_hub', 'innovation_hub_tech'
]

print(f"Features contínuas:\n{cont_features}")
print(f"Features binárias:\n{bin_features}")
print(f"Features categóricas:\n{cat_features}")

cont_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('cont', cont_transformer, cont_features),
    ('cat', cat_transformer, cat_features),
    ('bin', 'passthrough', bin_features)
])

X = train.drop(['id', 'labels'], axis=1)
y = train['labels']

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

print("\n========== RANDOM FOREST ==========")
model = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
rf_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
print(f"Acurácia validação: {model.score(X_val, y_val):.4f}")
print("Classif. Report:\n", classification_report(y_val, y_pred, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))
print(f"ROC-AUC validação: {rf_auc:.4f}")

# ===== Validação cruzada RF =====
rf_scores = cross_val_score(model, X, y, 
                            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
                            scoring='roc_auc')
print(f"Validação cruzada RandomForest ROC-AUC: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")

print("\n========== XGBOOST (Default) ==========")
xgb_pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=(y_train.value_counts()[0]/y_train.value_counts()[1])
    ))
])
xgb_pipeline.fit(X_train, y_train)
y_pred_xgb = xgb_pipeline.predict(X_val)
y_proba_xgb = xgb_pipeline.predict_proba(X_val)[:, 1]
xgb_auc = roc_auc_score(y_val, y_proba_xgb)
print(f"Acurácia validação: {xgb_pipeline.score(X_val, y_val):.4f}")
print("Classif. Report:\n", classification_report(y_val, y_pred_xgb, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_xgb))
print(f"ROC-AUC validação: {xgb_auc:.4f}")

# ===== Validação cruzada XGB default =====
xgb_scores = cross_val_score(xgb_pipeline, X, y, 
                             cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
                             scoring='roc_auc')
print(f"Validação cruzada XGBoost padrão ROC-AUC: {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}")

print("\n========== XGBOOST (GridSearchCV) ==========")
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [3, 5, 7],
    'clf__learning_rate': [0.01, 0.1, 0.2],
    'clf__subsample': [0.8, 1.0]
}
grid_xgb = GridSearchCV(
    xgb_pipeline,
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1
)
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_
y_pred_bestxgb = best_xgb.predict(X_val)
y_proba_bestxgb = best_xgb.predict_proba(X_val)[:, 1]
best_xgb_auc = roc_auc_score(y_val, y_proba_bestxgb)
print(f"Melhor ROC-AUC (cv): {grid_xgb.best_score_:.4f}")
print(f"Melhores parâmetros: {grid_xgb.best_params_}")
print(f"Acurácia validação: {best_xgb.score(X_val, y_val):.4f}")
print("Classif. Report:\n", classification_report(y_val, y_pred_bestxgb, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_bestxgb))
print(f"ROC-AUC validação: {best_xgb_auc:.4f}")

# ===== Validação cruzada XGB GridSearch =====
best_xgb_scores = cross_val_score(best_xgb, X, y, 
                                  cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
                                  scoring='roc_auc')
print(f"Validação cruzada XGBoost GridSearchCV ROC-AUC: {best_xgb_scores.mean():.4f} ± {best_xgb_scores.std():.4f}")

print("\n========== XGBOOST (RandomizedSearchCV) ==========")
param_dist = {
    'clf__n_estimators': [100, 200, 300, 400, 500],
    'clf__max_depth': [2, 3, 4, 5, 6, 7, 8],
    'clf__learning_rate': [0.005, 0.01, 0.05, 0.075, 0.1, 0.2, 0.3],
    'clf__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'clf__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'clf__gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'clf__reg_alpha': [0, 0.01, 0.1, 1, 10],
    'clf__reg_lambda': [0.01, 0.1, 1, 5, 10]
}
random_xgb = RandomizedSearchCV(
    xgb_pipeline,
    param_dist,
    n_iter=40,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)
random_xgb.fit(X_train, y_train)
print(f"Melhor ROC-AUC (cv): {random_xgb.best_score_:.4f}")
print(f"Melhores parâmetros: {random_xgb.best_params_}")
best_rnd_xgb = random_xgb.best_estimator_
y_pred_best_rndxgb = best_rnd_xgb.predict(X_val)
y_proba_best_rndxgb = best_rnd_xgb.predict_proba(X_val)[:, 1]
best_rnd_xgb_auc = roc_auc_score(y_val, y_proba_best_rndxgb)
print(f"Acurácia validação: {best_rnd_xgb.score(X_val, y_val):.4f}")
print("Classif. Report:\n", classification_report(y_val, y_pred_best_rndxgb, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_best_rndxgb))
print(f"ROC-AUC validação: {best_rnd_xgb_auc:.4f}")

# ===== Validação cruzada XGB RandomizedSearch =====
best_rnd_xgb_scores = cross_val_score(best_rnd_xgb, X, y, 
                                      cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
                                      scoring='roc_auc')
print(f"Validação cruzada XGBoost RandomizedSearchCV ROC-AUC: {best_rnd_xgb_scores.mean():.4f} ± {best_rnd_xgb_scores.std():.4f}")

print("\n========== GRADIENT BOOSTING ==========")

gb_pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', GradientBoostingClassifier(random_state=42))
])
gb_pipeline.fit(X_train, y_train)
y_pred_gb = gb_pipeline.predict(X_val)
y_proba_gb = gb_pipeline.predict_proba(X_val)[:, 1]
gb_auc = roc_auc_score(y_val, y_proba_gb)
print(f"Acurácia validação: {gb_pipeline.score(X_val, y_val):.4f}")
print("Classif. Report:\n", classification_report(y_val, y_pred_gb, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_gb))
print(f"ROC-AUC validação: {gb_auc:.4f}")

# ===== Validação cruzada Gradient Boosting =====
gb_scores = cross_val_score(gb_pipeline, X, y, 
                           cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
                           scoring='roc_auc')
print(f"Validação cruzada GradientBoosting ROC-AUC: {gb_scores.mean():.4f} ± {gb_scores.std():.4f}")


print("\n========== F1 THRESHOLD TUNING (RandomizedSearchCV) ==========")
thresholds = np.arange(0.3, 0.7, 0.01)
best_f1 = 0
best_th = 0.5
for th in thresholds:
    preds = (y_proba_best_rndxgb > th).astype(int)
    score = f1_score(y_val, preds)
    print(f"Threshold {th:.2f} -> F1: {score:.4f}")
    if score > best_f1:
        best_f1 = score
        best_th = th
print(f"\033[1mMelhor threshold para submissão (RandomizedSearchCV): {best_th:.2f} (F1={best_f1:.4f})\033[0m")

print("\n========== F1 THRESHOLD TUNING (GridSearchCV) ==========")
thresholds_grid = np.arange(0.3, 0.7, 0.01)
best_f1_grid = 0
best_th_grid = 0.5
for th in thresholds_grid:
    preds_grid = (y_proba_bestxgb > th).astype(int)
    score_grid = f1_score(y_val, preds_grid)
    print(f"Threshold {th:.2f} -> F1: {score_grid:.4f}")
    if score_grid > best_f1_grid:
        best_f1_grid = score_grid
        best_th_grid = th
print(f"\033[1mMelhor threshold para submissão (GridSearchCV): {best_th_grid:.2f} (F1={best_f1_grid:.4f})\033[0m")

# Gerar submissão usando o melhor threshold do GridSearchCV
X_test = test.drop(['id'], axis=1)
y_proba_test_bestxgb = best_xgb.predict_proba(X_test)[:, 1]
preds_test_bestxgb = (y_proba_test_bestxgb > best_th_grid).astype(int)
submission_xgb_threshold = pd.DataFrame({'id': test['id'], 'labels': preds_test_bestxgb})
submission_xgb_threshold.to_csv('submission_xgb_threshold.csv', index=False)


# Exemplo para obter importâncias do melhor modelo XGBoost
best_model = random_xgb.best_estimator_
preprocessor = best_model.named_steps['pre']
classifier = best_model.named_steps['clf']

# Obter nomes das features após o pré-processamento
cat_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_features)
all_feature_names = cont_features + list(cat_names) + bin_features

importances = pd.Series(classifier.feature_importances_, index=all_feature_names)
top_features = importances.sort_values(ascending=False).head(20)

# Plotar
plt.figure(figsize=(10, 8))
sns.barplot(x=top_features, y=top_features.index)
plt.title('Top 20 Feature Importances')
plt.show()

# Previsão final para submissão pode ser feita com o melhor modelo
X_test = test.drop(['id'], axis=1)
preds = best_xgb.predict(X_test)
submission = pd.DataFrame({'id': test['id'], 'labels': preds})
submission.to_csv('submission_grid_search.csv', index=False)