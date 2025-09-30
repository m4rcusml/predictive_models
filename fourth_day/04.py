import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import roc_auc_score, f1_score, classification_report

from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
import shap

def engineer_features(df):
  """
  Aplica todas as transformações de feature engineering a um dataset
  """
  df_enhanced = df.copy()
  
  # 1. Features de proporções e ratios
  df_enhanced['milestones_per_funding_round'] = df_enhanced['milestones'] / (df_enhanced['funding_rounds'] + 0.1)
  df_enhanced['avg_funding_per_round'] = df_enhanced['funding_total_usd'] / (df_enhanced['funding_rounds'] + 0.1)
  df_enhanced['relationships_per_milestone'] = df_enhanced['relationships'] / (df_enhanced['milestones'] + 0.1)
  
  # 2. Features baseadas em tempo
  df_enhanced['time_to_first_funding'] = df_enhanced['age_first_funding_year'].fillna(999)
  df_enhanced['funding_duration'] = (df_enhanced['age_last_funding_year'] - df_enhanced['age_first_funding_year']).fillna(0)
  df_enhanced['time_to_first_milestone'] = df_enhanced['age_first_milestone_year'].fillna(999)
  df_enhanced['milestone_duration'] = (df_enhanced['age_last_milestone_year'] - df_enhanced['age_first_milestone_year']).fillna(0)
  
  # 3. Features de progressão de funding
  df_enhanced['funding_progression_score'] = (
      df_enhanced['has_roundA'] * 1 +
      df_enhanced['has_roundB'] * 2 +
      df_enhanced['has_roundC'] * 3 +
      df_enhanced['has_roundD'] * 4
  )
  df_enhanced['funding_diversity'] = df_enhanced['has_VC'] + df_enhanced['has_angel']
  df_enhanced['has_later_stage'] = ((df_enhanced['has_roundC'] == 1) | (df_enhanced['has_roundD'] == 1)).astype(int)
  
  # 4. Features geográficas
  df_enhanced['location_success_score'] = (
      df_enhanced['is_MA'] * 0.82 +
      df_enhanced['is_NY'] * 0.70 +
      df_enhanced['is_CA'] * 0.69 +
      df_enhanced['is_TX'] * 0.46 +
      df_enhanced['is_otherstate'] * 0.46
  )
  df_enhanced['is_top_tier_location'] = ((df_enhanced['is_CA'] == 1) | 
                                        (df_enhanced['is_NY'] == 1) | 
                                        (df_enhanced['is_MA'] == 1)).astype(int)
  
  # 5. Features setoriais
  df_enhanced['sector_success_score'] = (
      df_enhanced['is_enterprise'] * 0.75 +
      df_enhanced['is_advertising'] * 0.69 +
      df_enhanced['is_web'] * 0.68 +
      df_enhanced['is_biotech'] * 0.68 +
      df_enhanced['is_mobile'] * 0.66 +
      df_enhanced['is_software'] * 0.64 +
      df_enhanced['is_gamesvideo'] * 0.62 +
      df_enhanced['is_othercategory'] * 0.62 +
      df_enhanced['is_consulting'] * 0.50 +
      df_enhanced['is_ecommerce'] * 0.40
  )
  df_enhanced['is_high_success_sector'] = ((df_enhanced['is_enterprise'] == 1) | 
                                          (df_enhanced['is_advertising'] == 1) | 
                                          (df_enhanced['is_web'] == 1) | 
                                          (df_enhanced['is_biotech'] == 1)).astype(int)
  
  # 6. Features de escala e maturidade
  df_enhanced['total_activity_score'] = (
      df_enhanced['relationships'] + 
      df_enhanced['milestones'] + 
      df_enhanced['funding_rounds']
  )
  df_enhanced['funding_efficiency'] = df_enhanced['relationships'] / (df_enhanced['funding_total_usd'] / 1000000 + 0.1)
  df_enhanced['is_mature_startup'] = ((df_enhanced['milestones'] >= 2) & 
                                      (df_enhanced['funding_rounds'] >= 2)).astype(int)
  
  # 7. Features de interação
  df_enhanced['location_tech_interaction'] = (
      df_enhanced['is_top_tier_location'] * 
      (df_enhanced['is_software'] + df_enhanced['is_web'] + df_enhanced['is_mobile'])
  )
  df_enhanced['funding_relationships_interaction'] = (
      df_enhanced['funding_progression_score'] * df_enhanced['relationships']
  )
  df_enhanced['maturity_funding_interaction'] = (
      df_enhanced['is_mature_startup'] * np.log1p(df_enhanced['funding_total_usd'])
  )
  df_enhanced['has_recent_milestones'] = (
      (df_enhanced['age_last_milestone_year'].fillna(999) <= 3)
  ).astype(int)
  
  return df_enhanced

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Remove colunas não usadas
for col in ['id', 'category_code']:
    if col in train.columns:
        train = train.drop(columns=[col])
    if col in test.columns:
        test = test.drop(columns=[col])

# Feature Engineering
X_train_enhanced = engineer_features(train.drop(columns=['labels']))
X_test_enhanced = engineer_features(test)

# Recoloca target
y_train = train['labels']

# Tratamento de missing (exceto onde NaN já representa "não ocorreu")
for col in X_train_enhanced.columns:
    if X_train_enhanced[col].dtype in [np.float64, np.int64]:
        mediana = X_train_enhanced[col].median()
        X_train_enhanced.fillna({col: mediana}, inplace=True)
        X_test_enhanced.fillna({col: mediana}, inplace=True)

# Tratamento de outliers (tree-based: clipping nos percentis extremos)
for col in X_train_enhanced.columns:
    if X_train_enhanced[col].dtype in [np.float64, np.int64]:
        lower = X_train_enhanced[col].quantile(0.25)
        upper = X_train_enhanced[col].quantile(0.75)
        X_train_enhanced[col] = X_train_enhanced[col].clip(lower, upper)
        X_test_enhanced[col] = X_test_enhanced[col].clip(lower, upper)

# Normalização (opcional para tree-based, mas pode ajudar em análise)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_enhanced)
X_test_scaled = scaler.transform(X_test_enhanced)

# --- Análise Exploratória ---

# Labels
print("Distribuição do target (labels):")
print(y_train.value_counts(normalize=True))
sns.countplot(x=y_train)
plt.title("Distribuição dos Labels (Sucesso vs Fracasso)")
plt.xlabel("Label (0: Fracasso, 1: Sucesso)")
plt.ylabel("Quantidade")
# plt.show()

# Estatísticas
print("\nResumo estatístico geral das features:")
print(pd.DataFrame(X_train_enhanced).describe().T)

# Boxplots das principais features
top_features_to_plot = X_train_enhanced.columns[:7]
for feature in top_features_to_plot:
    plt.figure()
    sns.boxplot(x=y_train, y=X_train_enhanced[feature])
    plt.title(f"Boxplot de {feature} por label")
    plt.xlabel("Label")
    plt.ylabel(feature)
    # plt.show()

# Multicolinearidade
corr_matrix = X_train_enhanced.corr().abs()
plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title("Matriz de Correlação entre Features")
# plt.show()

# Features altamente correlacionadas (> 0.8)
high_corr_var = np.where(corr_matrix > 0.8)
high_corr_list = [
  (X_train_enhanced.columns[x], X_train_enhanced.columns[y], corr_matrix.iloc[x, y]) 
  for x, y in zip(*high_corr_var) if x != y and x < y
]
print("\nPares de features altamente correlacionadas (> 0.8):")
for var1, var2, corr in high_corr_list:
  print(f"{var1} & {var2}: correlação = {corr:.3f}")



# Seleção de Features usando SelectKBest
# selector = SelectKBest(score_func=f_classif, k=15)
# selector.fit(X_train_enhanced, y_train)

# # Features selecionadas
# feature_mask = selector.get_support()
# selected_features = X_train_enhanced.columns[feature_mask]
# X_train_selected = X_train_enhanced[selected_features]
# X_test_selected = X_test_enhanced[selected_features]


# Seleção de features manual
# Seleção manual apenas das features criadas pela função engineer_features
engineered_feature_names = [
    # Core features (must-have)
    'total_activity_score',
    'maturity_funding_interaction', 
    'funding_progression_score',
    'is_mature_startup',
    'time_to_first_milestone',
    'funding_relationships_interaction',
    
    # Supporting features (high value)
    'milestone_duration',
    'location_success_score',
    'funding_duration',
    'is_top_tier_location',
    'has_later_stage',
    
    # Efficiency features
    'milestones_per_funding_round',
    'funding_efficiency'
]

X_train_selected = X_train_enhanced[engineered_feature_names]
X_test_selected = X_test_enhanced[engineered_feature_names]


# Split para validação (usando stratificação para balanceamento)
X_train, X_val, y_train_split, y_val = train_test_split(
  X_train_selected, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# Modelagem - exemplo com Logistic Regression e Random Forest
models = {
  'RandomForest': RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
  'ExtraTrees': ExtraTreesClassifier(class_weight='balanced', n_estimators=100, random_state=42),
  'XGBClassifier': XGBClassifier(class_weight='balanced', n_estimators=100, random_state=42)
}

# === AVALIAÇÃO AVANÇADA ===

for name, model in models.items():
    model.fit(X_train, y_train_split)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    print(f"\nModelo: {name}")
    print(f"AUC-ROC: {roc_auc_score(y_val, y_prob):.4f}")
    print(f"F1-Score: {f1_score(y_val, y_pred):.4f}")
    print(classification_report(y_val, y_pred))

    # Confusion Matrix
    conf_mat = confusion_matrix(y_val, y_pred)
    print("Matriz de Confusão:\n", conf_mat)
    plt.figure()
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matriz de Confusão - {name}")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    # plt.show()

    # Curva ROC
    fpr, tpr, thresholds = roc_curve(y_val, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_val, y_prob):.2f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curva ROC - {name}')
    plt.legend()
    # plt.show()

    # Curva Precision-Recall
    precision, recall, thresholds = precision_recall_curve(y_val, y_prob)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Curva Precision-Recall - {name}')
    # plt.show()

# === OTIMIZAÇÃO DE THRESHOLD ===
# Exemplo para o último modelo treinado:
optimal_threshold = 0.5  # Você pode ajustar esse valor após analisar as curvas
optimized_pred = (y_prob >= optimal_threshold).astype(int)
print(f"F1-Score com threshold {optimal_threshold}: {f1_score(y_val, optimized_pred):.4f}")

# === INTERPRETAÇÃO DOS RESULTADOS ===
# (Exemplo com SHAP para árvore)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val, feature_names=X_val.columns)

# Feature Importance tradicional:
importances = model.feature_importances_
plt.figure()
sns.barplot(x=importances, y=X_val.columns)
plt.title(f"Importância das Features - {name}")
# plt.show()

# === PREVISÃO NO TEST SET ===
test_pred_prob = model.predict_proba(X_test_selected)[:, 1]
test_pred_label = (test_pred_prob >= optimal_threshold).astype(int)

# Para submissão/pipeline final
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission['labels'] = test_pred_label
sample_submission.to_csv('submission.csv', index=False)
