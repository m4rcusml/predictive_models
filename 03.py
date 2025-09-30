import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import category_encoders as ce
from sklearn.feature_selection import SelectFromModel

# ========== LEITURA E FEATURE ENGINEERING ===========
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

for df in [train, test]:
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

cont_features = [
    'age_first_funding_year', 'age_last_funding_year',
    'age_first_milestone_year', 'age_last_milestone_year',
    'relationships', 'funding_rounds', 'funding_total_usd',
    'milestones', 'avg_participants', 'relationships_per_round',
    'participants_per_round', 'funding_per_year', 'milestones_per_year',
    'funding_per_round', 'time_to_first_milestone_after_funding', 'milestone_per_million'
]
bin_features = [col for col in train.columns if col.startswith('is_') or col.startswith('has_') and col != 'is_othercategory']
cat_features = ['category_code']

# ========== TARGET ENCODER ===========
# (apenas para category_code)
all_features = cont_features + bin_features + ['category_code']
X = train[all_features].copy()
y = train['labels']
X_test = test[all_features].copy()

target_encoder = ce.TargetEncoder(cols=['category_code'])
X['category_code'] = target_encoder.fit_transform(X['category_code'], y)
X_test['category_code'] = target_encoder.transform(X_test['category_code'])

# ========== PRE-PROCESSAMENTO ===========
cont_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
preprocessor = ColumnTransformer([
    ('cont', cont_transformer, cont_features),
    ('bin', 'passthrough', bin_features),
    ('cat', 'passthrough', ['category_code'])
])

X_prep = preprocessor.fit_transform(X)
X_test_prep = preprocessor.transform(X_test)

# ========== FEATURE SELECTION COM XGBOOST ===========
print("\n===== FEATURE SELECTION (XGBoost) =====")
feature_selector_clf = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=(y.value_counts()[0]/y.value_counts()[1])
)
feature_selector_clf.fit(X_prep, y)
selector = SelectFromModel(feature_selector_clf, prefit=True, threshold='median')
X_fs = selector.transform(X_prep)
X_test_fs = selector.transform(X_test_prep)

print(f"Nº features selecionadas: {X_fs.shape[1]}/{X_prep.shape[1]}")

# Split para validação
X_train, X_val, y_train, y_val = train_test_split(X_fs, y, stratify=y, test_size=0.2, random_state=42)

# ========== XGBOOST - RANDOM SEARCH ===========
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [2, 3, 4, 5, 6, 7],
    'learning_rate': [0.005, 0.01, 0.05, 0.075, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],
    'reg_alpha': [0, 0.01, 0.1, 1, 10],
    'reg_lambda': [0.01, 0.1, 1, 5, 10]
}
random_xgb = RandomizedSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss',
                  scale_pos_weight=(y_train.value_counts()[0]/y_train.value_counts()[1])),
    param_dist,
    n_iter=40,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
random_xgb.fit(X_train, y_train)
best_xgb = random_xgb.best_estimator_

print("\n===== XGBoost RANDOMIZED SEARCH =====")
print("Melhores parâmetros:", random_xgb.best_params_)
print(f"Acurácia validação: {best_xgb.score(X_val, y_val):.3f}")
y_pred_val = best_xgb.predict(X_val)
y_proba_val = best_xgb.predict_proba(X_val)[:,1]
print("CLASSIFICATION REPORT:\n", classification_report(y_val, y_pred_val))
print("CONFUSION MATRIX:\n", confusion_matrix(y_val, y_pred_val))
print(f'ROC-AUC: {roc_auc_score(y_val, y_proba_val):.4f}')

xgb_cv = cross_val_score(best_xgb, X_fs, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='roc_auc')
print(f"Validação cruzada FINAL XGBoost (RandomizedSearchCV) ROC-AUC: {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")

# ========== THRESHOLD TUNING ADICIONAL ===========
print("\n===== THRESHOLD TUNING (XGBoost) =====")
thresholds = np.arange(0.2, 0.7, 0.005)
f1s = []
for th in thresholds:
    preds = (y_proba_val > th).astype(int)
    f1 = f1_score(y_val, preds)
    f1s.append(f1)
best_f1_idx = np.argmax(f1s)
best_th = thresholds[best_f1_idx]
print(f"Melhor threshold para submissão: {best_th:.3f} (F1 = {f1s[best_f1_idx]:.4f})")

# ========== SUBMISSÃO ===========
final_proba_test = best_xgb.predict_proba(X_test_fs)[:,1]
final_preds = (final_proba_test > best_th).astype(int)
submission = pd.DataFrame({'id': test['id'], 'labels': final_preds})
submission.to_csv('submission_selected_encoded.csv', index=False)
print("\nArquivo gerado: submission_selected_encoded.csv")
