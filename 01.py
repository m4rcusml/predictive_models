import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

for df in [train, test]:
    # Funding por rodada
    df['funding_per_round'] = df['funding_total_usd'] / (df['funding_rounds'] + 1e-9)
    # Milestones por cada ano da idade do último milestone (preenche NaN com mediana)
    df['milestones_per_year'] = df['milestones'] / (df['age_last_milestone_year'].fillna(df['age_last_milestone_year'].median()) + 1)
    # Funding dividido pela idade de captação final
    df['funding_per_year'] = df['funding_total_usd'] / (df['age_last_funding_year'].fillna(df['age_last_funding_year'].median()) + 1)
    # Relação de investidores por rodada
    df['participants_per_round'] = df['avg_participants'] / (df['funding_rounds'] + 1e-9)
    # Relação de networking por rodada
    df['relationships_per_round'] = df['relationships'] / (df['funding_rounds'] + 1e-9)
    # Interações de dummies regionais/setoriais
    df['is_CA_software'] = df['is_CA'] * df['is_software']
    df['is_NY_advertising'] = df['is_NY'] * df['is_advertising']
    df['is_MA_biotech'] = df['is_MA'] * df['is_biotech']
    # Feature binária por faixas de funding (>mediana)
    df['is_high_funding'] = (df['funding_total_usd'] > df['funding_total_usd'].median()).astype(int)
    # Feature binária por faixas de milestones (>mediana)
    df['is_high_milestones'] = (df['milestones'] > df['milestones'].median()).astype(int)
    df['time_to_first_milestone_after_funding'] = (
        df['age_first_milestone_year'] - df['age_first_funding_year']
    )
    df['milestone_per_million'] = df['milestones'] / (df['funding_total_usd'] / 1e6 + 1)

# Separando features
cont_features = [
    'age_first_funding_year', 'age_last_funding_year',
    'age_first_milestone_year', 'age_last_milestone_year',
    'relationships', 'funding_rounds', 'funding_total_usd',
    'milestones', 'avg_participants', 'relationships_per_round',
    'participants_per_round', 'funding_per_year', 'milestones_per_year',
    'funding_per_round'
]
cat_features = ['category_code']
bin_features = [col for col in train.columns if col.startswith('is_') or col.startswith('has_') and col != 'is_othercategory']

# Imputação + Escalonamento para contínuas
cont_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
# One-Hot para categórica
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

model = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
])

model.fit(X_train, y_train)
score = model.score(X_val, y_val)
print(f'Acurácia no conjunto de validação: {score:.3f}')
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))
print(confusion_matrix(y_val, y_pred))
print('ROC-AUC:', roc_auc_score(y_val, model.predict_proba(X_val)[:,1]))

# XGBoost pipeline SEM hiperparâmetros ainda
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

print('--- XGBoost ---')
print(f'Acurácia: {xgb_pipeline.score(X_val, y_val):.3f}')
print(classification_report(y_val, y_pred_xgb))
print(confusion_matrix(y_val, y_pred_xgb))
print('ROC-AUC:', roc_auc_score(y_val, y_proba_xgb))

# AJUSTE DE HIPERPARÂMETROS COM GRIDSEARCHCV
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

print('--- XGBoost com hiperparâmetros ótimos ---')
print('Melhor ROC-AUC (cv):', grid_xgb.best_score_)
print('Melhores parâmetros:', grid_xgb.best_params_)
best_xgb = grid_xgb.best_estimator_
print(f'Acurácia validação: {best_xgb.score(X_val, y_val):.3f}')
y_pred_bestxgb = best_xgb.predict(X_val)
y_proba_bestxgb = best_xgb.predict_proba(X_val)[:, 1]
print(classification_report(y_val, y_pred_bestxgb))
print(confusion_matrix(y_val, y_pred_bestxgb))
print('ROC-AUC:', roc_auc_score(y_val, y_proba_bestxgb))

from sklearn.model_selection import RandomizedSearchCV

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
    n_iter=40,              # Número de combinações a testar (aumente para busca mais ampla)
    cv=3,
    scoring='roc_auc',      # Pode trocar para 'f1' se quiser priorizar acerto de positivos
    n_jobs=-1,
    random_state=42
)

random_xgb.fit(X_train, y_train)

print('--- XGBoost com RandomizedSearch ---')
print('Melhor ROC-AUC (cv):', random_xgb.best_score_)
print('Melhores parâmetros:', random_xgb.best_params_)

best_rnd_xgb = random_xgb.best_estimator_
print(f'Acurácia validação: {best_rnd_xgb.score(X_val, y_val):.3f}')
y_pred_best_rndxgb = best_rnd_xgb.predict(X_val)
y_proba_best_rndxgb = best_rnd_xgb.predict_proba(X_val)[:, 1]
print(classification_report(y_val, y_pred_best_rndxgb))
print(confusion_matrix(y_val, y_pred_best_rndxgb))
print('ROC-AUC:', roc_auc_score(y_val, y_proba_best_rndxgb))

# Validação cruzada para melhor modelo encontrado
best_rnd_xgb_scores = cross_val_score(
    best_rnd_xgb, X, y, 
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
    scoring='roc_auc'
)
print(f'Validação cruzada XGBoost (random search) ROC-AUC: {best_rnd_xgb_scores.mean():.3f} ± {best_rnd_xgb_scores.std():.3f}')


# --- Validação cruzada Random Forest ---
rf_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='roc_auc')
print(f'Validação cruzada RandomForest ROC-AUC: {rf_scores.mean():.3f} ± {rf_scores.std():.3f}')

# --- Validação cruzada XGBoost ---
xgb_scores = cross_val_score(xgb_pipeline, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='roc_auc')
print(f'Validação cruzada XGBoost ROC-AUC: {xgb_scores.mean():.3f} ± {xgb_scores.std():.3f}')

# --- Validação cruzada XGBoost hiperparametrizado ---
best_xgb_scores = cross_val_score(best_xgb, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='roc_auc')
print(f'Validação cruzada XGBoost (melhor modelo) ROC-AUC: {best_xgb_scores.mean():.3f} ± {best_xgb_scores.std():.3f}')


# Previsão final para submissão pode ser feita com o melhor modelo
X_test = test.drop(['id'], axis=1)
preds = best_rnd_xgb.predict(X_test)
submission = pd.DataFrame({'id': test['id'], 'labels': preds})
submission.to_csv('submission.csv', index=False)


# the best public score are from the xgboost default model and xgboost with random search models