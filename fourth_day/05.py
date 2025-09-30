# ============================================================================
# MODELO PREDITIVO DE SUCESSO DE STARTUPS - VERSÃO MELHORADA
# Baseado em pesquisa acadêmica e melhores práticas de machine learning
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Imports para ML
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFECV, SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Balanceamento de classes
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Métricas
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report, 
    confusion_matrix, precision_recall_curve, roc_curve, 
    average_precision_score, precision_score, recall_score
)

# Interpretabilidade
import shap

def engineer_features_improved(df):
    """
    Feature engineering melhorado baseado em literatura acadêmica
    e análise dos gaps identificados no código original
    """
    df_enhanced = df.copy()
    
    # ================================================================
    # 1. FEATURES ORIGINAIS MANTIDAS (bem alinhadas com literatura)
    # ================================================================
    
    # Features de proporções e ratios (MANTIDAS - muito boas)
    df_enhanced['milestones_per_funding_round'] = df_enhanced['milestones'] / (df_enhanced['funding_rounds'] + 0.1)
    df_enhanced['avg_funding_per_round'] = df_enhanced['funding_total_usd'] / (df_enhanced['funding_rounds'] + 0.1)
    df_enhanced['relationships_per_milestone'] = df_enhanced['relationships'] / (df_enhanced['milestones'] + 0.1)
    
    # Features temporais (MANTIDAS - inovadoras)
    df_enhanced['time_to_first_funding'] = df_enhanced['age_first_funding_year'].fillna(999)
    df_enhanced['funding_duration'] = (df_enhanced['age_last_funding_year'] - df_enhanced['age_first_funding_year']).fillna(0)
    df_enhanced['time_to_first_milestone'] = df_enhanced['age_first_milestone_year'].fillna(999)
    df_enhanced['milestone_duration'] = (df_enhanced['age_last_milestone_year'] - df_enhanced['age_first_milestone_year']).fillna(0)
    
    # Features de progressão (MANTIDAS - bem fundamentadas)
    df_enhanced['funding_progression_score'] = (
        df_enhanced['has_roundA'] * 1 +
        df_enhanced['has_roundB'] * 2 + 
        df_enhanced['has_roundC'] * 3 +
        df_enhanced['has_roundD'] * 4
    )
    df_enhanced['funding_diversity'] = df_enhanced['has_VC'] + df_enhanced['has_angel']
    df_enhanced['has_later_stage'] = ((df_enhanced['has_roundC'] == 1) | (df_enhanced['has_roundD'] == 1)).astype(int)
    
    # ================================================================
    # 2. MELHORIAS NAS FEATURES GEOGRÁFICAS E SETORIAIS
    # ================================================================
    
    # Features geográficas melhoradas (sem hard-coding)
    df_enhanced['is_innovation_hub'] = (
        (df_enhanced['is_CA'] == 1) | 
        (df_enhanced['is_NY'] == 1) | 
        (df_enhanced['is_MA'] == 1)
    ).astype(int)
    
    df_enhanced['is_secondary_hub'] = (df_enhanced['is_TX'] == 1).astype(int)
    
    # Features setoriais melhoradas
    df_enhanced['is_tech_sector'] = (
        (df_enhanced['is_software'] == 1) |
        (df_enhanced['is_web'] == 1) |
        (df_enhanced['is_mobile'] == 1)
    ).astype(int)
    
    df_enhanced['is_high_capital_sector'] = (
        (df_enhanced['is_biotech'] == 1) |
        (df_enhanced['is_enterprise'] == 1)
    ).astype(int)
    
    df_enhanced['is_consumer_facing'] = (
        (df_enhanced['is_ecommerce'] == 1) |
        (df_enhanced['is_advertising'] == 1) |
        (df_enhanced['is_gamesvideo'] == 1)
    ).astype(int)
    
    # ================================================================
    # 3. NOVAS FEATURES BASEADAS NA LITERATURA (GAPS IDENTIFICADOS)
    # ================================================================
    
    # A. TEAM & NETWORK FEATURES (Gap crítico - score 3/10)
    df_enhanced['network_density'] = df_enhanced['relationships'] / (df_enhanced['funding_rounds'] + 1)
    df_enhanced['relationship_efficiency'] = df_enhanced['milestones'] / (df_enhanced['relationships'] + 0.1)
    df_enhanced['team_network_strength'] = df_enhanced['relationships'] * df_enhanced['milestones'] / 10  # Proxy para força da rede
    df_enhanced['founder_experience_proxy'] = np.minimum(df_enhanced['relationships'] / 5, 3)  # Proxy baseado em network
    df_enhanced['team_size_proxy'] = np.log1p(df_enhanced['relationships']) / 2  # Proxy para tamanho da equipe
    
    # B. FINANCIAL EFFICIENCY FEATURES (Melhoramento do Gap Operacional)
    df_enhanced['capital_efficiency'] = df_enhanced['milestones'] / (df_enhanced['funding_total_usd'] / 1000000 + 0.1)
    df_enhanced['funding_momentum'] = df_enhanced['funding_rounds'] / (df_enhanced['age_last_funding_year'] + 0.1)
    df_enhanced['funding_velocity'] = df_enhanced['funding_total_usd'] / (df_enhanced['age_last_funding_year'] + 0.1)
    df_enhanced['burn_rate_proxy'] = df_enhanced['funding_total_usd'] / (df_enhanced['funding_duration'] + 0.1)
    df_enhanced['runway_efficiency'] = df_enhanced['milestones'] / (df_enhanced['burn_rate_proxy'] + 0.1)
    
    # C. PRODUCT-MARKET FIT PROXIES (Gap crítico - score 4/10)
    df_enhanced['traction_proxy'] = df_enhanced['milestones'] / (df_enhanced['time_to_first_funding'] + 1)
    df_enhanced['development_speed'] = df_enhanced['milestones'] / (df_enhanced['funding_rounds'] + 0.1)
    df_enhanced['market_validation_score'] = (
        (df_enhanced['milestones'] > 0).astype(int) +
        (df_enhanced['funding_rounds'] > 1).astype(int) +
        (df_enhanced['has_VC'] == 1).astype(int)
    )
    df_enhanced['customer_traction_proxy'] = df_enhanced['relationships'] * df_enhanced['milestones'] / (df_enhanced['funding_rounds'] + 1)
    df_enhanced['product_maturity'] = np.minimum(df_enhanced['milestone_duration'] / 2, 5)  # Anos de desenvolvimento
    
    # D. RISK INDICATORS (Novo conjunto crítico)
    df_enhanced['funding_concentration_risk'] = df_enhanced['avg_funding_per_round'] / (df_enhanced['funding_total_usd'] + 0.1)
    df_enhanced['milestone_stagnation_risk'] = (df_enhanced['age_last_milestone_year'].fillna(999) > 3).astype(int)
    df_enhanced['late_funding_risk'] = (df_enhanced['time_to_first_funding'] > 2).astype(int)
    df_enhanced['funding_gap_risk'] = np.maximum(0, df_enhanced['age_last_funding_year'] - 2)  # Anos desde último funding
    df_enhanced['low_activity_risk'] = (
        (df_enhanced['milestones'] == 0) | 
        (df_enhanced['relationships'] < 3)
    ).astype(int)
    
    # E. MATURITY & SCALE INDICATORS (Melhorados)
    df_enhanced['startup_maturity_score'] = (
        (df_enhanced['milestones'] >= 1).astype(int) +
        (df_enhanced['funding_rounds'] >= 2).astype(int) +
        (df_enhanced['relationships'] >= 5).astype(int) +
        (df_enhanced['has_later_stage'] == 1).astype(int) * 2
    )
    
    df_enhanced['scale_readiness'] = (
        (df_enhanced['funding_progression_score'] >= 2) &
        (df_enhanced['milestones'] >= 2) &
        (df_enhanced['relationships'] >= 3)
    ).astype(int)
    
    df_enhanced['growth_stage_indicator'] = np.minimum(
        df_enhanced['funding_progression_score'] + df_enhanced['startup_maturity_score'], 8
    )
    
    # F. TIMING & COMPETITIVE FEATURES
    df_enhanced['optimal_funding_timing'] = (
        (df_enhanced['time_to_first_funding'] >= 0.5) & 
        (df_enhanced['time_to_first_funding'] <= 2.0)
    ).astype(int)
    
    df_enhanced['funding_consistency'] = 1 / (df_enhanced['funding_duration'] + 0.1)
    df_enhanced['competitive_positioning'] = (
        df_enhanced['is_innovation_hub'].astype(int) * 2 +
        df_enhanced['is_tech_sector'].astype(int) +
        (df_enhanced['funding_total_usd'] > df_enhanced['funding_total_usd'].median()).astype(int)
    )
    
    # G. INTERACTION FEATURES MELHORADAS
    df_enhanced['location_funding_synergy'] = (
        df_enhanced['is_innovation_hub'] * np.log1p(df_enhanced['funding_total_usd'])
    )
    
    df_enhanced['sector_funding_fit'] = (
        df_enhanced['is_high_capital_sector'] * df_enhanced['funding_progression_score']
    )
    
    df_enhanced['team_market_fit'] = df_enhanced['network_density'] * df_enhanced['market_validation_score']
    df_enhanced['execution_capability'] = df_enhanced['development_speed'] * df_enhanced['capital_efficiency']
    
    # H. ADVANCED COMPOSITE FEATURES
    df_enhanced['overall_health_score'] = (
        df_enhanced['startup_maturity_score'] * 0.3 +
        df_enhanced['market_validation_score'] * 0.4 +
        df_enhanced['competitive_positioning'] * 0.3
    )
    
    df_enhanced['investment_attractiveness'] = (
        df_enhanced['funding_progression_score'] * 0.4 +
        df_enhanced['capital_efficiency'] * 0.3 +
        df_enhanced['network_density'] * 0.3
    )

    # Timing e recency (crítico para startups)
    df_enhanced['funding_recency'] = 10 - df_enhanced['age_last_funding_year'].fillna(10)
    df_enhanced['activity_recency_score'] = df_enhanced['funding_recency'] + df_enhanced['milestone_recency']

    # Quality indicators  
    df_enhanced['high_quality_network'] = (df_enhanced['relationships'] > df_enhanced['relationships'].quantile(0.75)).astype(int)
    df_enhanced['success_signals'] = df_enhanced['has_later_stage'] + df_enhanced['is_innovation_hub'] + df_enhanced['high_quality_network'] + df_enhanced['rapid_development'] + df_enhanced['is_well_funded']
    
    return df_enhanced


def advanced_preprocessing_pipeline(X_train, y_train, X_test):
    """
    Pipeline avançado de pré-processamento
    """
    print("=== INICIANDO PIPELINE AVANÇADO ===")
    
    # 1. Tratamento inteligente de missing values
    print("1. Tratamento de missing values...")
    
    # Para features temporais, NaN significa "evento não ocorreu"
    temporal_features = ['age_first_funding_year', 'age_last_funding_year', 
                        'age_first_milestone_year', 'age_last_milestone_year']
    
    for col in X_train.columns:
        if col in temporal_features:
            # Para temporais, manter NaN como valor alto (evento não ocorreu)
            continue
        elif X_train[col].dtype in [np.float64, np.int64]:
            # Para outras numéricas, usar mediana
            median_val = X_train[col].median()
            X_train[col].fillna(median_val, inplace=True)
            X_test[col].fillna(median_val, inplace=True)
    
    # 2. Tratamento de outliers mais inteligente
    print("2. Tratamento de outliers...")
    
    # Lista de features onde outliers podem ser informativos
    preserve_outliers = ['funding_total_usd', 'relationships', 'milestones']
    
    for col in X_train.columns:
        if X_train[col].dtype in [np.float64, np.int64] and col not in preserve_outliers:
            # Usar IQR mais conservador
            Q1 = X_train[col].quantile(0.05)
            Q3 = X_train[col].quantile(0.95)
            X_train[col] = X_train[col].clip(Q1, Q3)
            X_test[col] = X_test[col].clip(Q1, Q3)
    
    print("3. Pipeline de preprocessing concluído!")
    return X_train, X_test


def create_advanced_ml_pipeline():
    """
    Cria pipeline de ML avançado baseado nas melhores práticas
    """
    
    # 1. Feature Selection Automática
    feature_selector = RFECV(
        estimator=RandomForestClassifier(n_estimators=50, random_state=42),
        step=1,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),  # Reduzido para eficiência
        scoring='roc_auc',
        min_features_to_select=15,
        n_jobs=-1
    )
    
    # 2. Balanceamento com SMOTE
    smote = SMOTE(random_state=42, k_neighbors=3)
    
    # 3. Scaler robusto
    scaler = RobustScaler()
    
    # 4. Ensemble de modelos diversos
    models = {
        'rf': RandomForestClassifier(
            n_estimators=150, 
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'xgb': XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ),
        'lgb': LGBMClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
    }
    
    # 5. Voting Ensemble
    ensemble = VotingClassifier(
        estimators=list(models.items()),
        voting='soft',
        n_jobs=-1
    )
    
    # 6. Pipeline completo
    pipeline = ImbPipeline([
      ('scaler', scaler),
      ('imputer', SimpleImputer(strategy='median')),
      ('feature_selection', feature_selector),
      ('smote', smote),
      ('ensemble', ensemble)
    ])
    
    return pipeline


def comprehensive_model_evaluation(pipeline, X_train, y_train, X_val, y_val):
    """
    Avaliação abrangente do modelo
    """
    print("\n=== AVALIAÇÃO ABRANGENTE DO MODELO ===")
    
    # Treinar o pipeline
    print("Treinando pipeline completo...")
    pipeline.fit(X_train, y_train)
    
    # Predições
    y_pred = pipeline.predict(X_val)
    y_prob = pipeline.predict_proba(X_val)[:, 1]
    
    # Métricas principais
    auc_roc = roc_auc_score(y_val, y_prob)
    auc_pr = average_precision_score(y_val, y_prob)
    f1 = f1_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    
    print(f"\nMÉTRICAS PRINCIPAIS:")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR: {auc_pr:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Relatório detalhado
    print(f"\nRELATÓRIO DETALHADO:")
    print(classification_report(y_val, y_pred))
    
    # Matriz de confusão
    conf_matrix = confusion_matrix(y_val, y_pred)
    print(f"\nMATRIZ DE CONFUSÃO:")
    print(conf_matrix)
    
    # Otimização de threshold
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_val, y_prob)
    f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\nOTIMIZAÇÃO DE THRESHOLD:")
    print(f"Threshold ótimo: {optimal_threshold:.3f}")
    print(f"F1-Score ótimo: {f1_scores[optimal_idx]:.4f}")
    
    return pipeline, optimal_threshold, {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'optimal_threshold': optimal_threshold
    }


def feature_importance_analysis(pipeline, feature_names):
    """
    Análise de importância das features
    """
    print("\n=== ANÁLISE DE IMPORTÂNCIA DAS FEATURES ===")
    
    try:
        # Pegar o modelo ensemble do pipeline
        ensemble = pipeline.named_steps['ensemble']
        
        # Feature importance média dos modelos
        importances = np.zeros(len(feature_names))
        
        for name, model in ensemble.estimators_:
            if hasattr(model, 'feature_importances_'):
                importances += model.feature_importances_
        
        importances /= len(ensemble.estimators_)
        
        # Criar dataframe para visualização
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("TOP 15 FEATURES MAIS IMPORTANTES:")
        print(feature_importance_df.head(15))
        
        return feature_importance_df
        
    except Exception as e:
        print(f"Erro na análise de importância: {e}")
        return None


def cross_validation_analysis(pipeline, X, y, cv_folds=5):
    """
    Análise de validação cruzada
    """
    print(f"\n=== VALIDAÇÃO CRUZADA ({cv_folds} FOLDS) ===")
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Múltiplas métricas
    scoring = ['roc_auc', 'f1', 'precision', 'recall']
    
    results = {}
    for score in scoring:
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=score, n_jobs=-1)
        results[score] = scores
        print(f"{score.upper()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return results


def main_pipeline():
    """
    Pipeline principal de execução
    """
    print("=== MODELO PREDITIVO DE SUCESSO DE STARTUPS - VERSÃO MELHORADA ===\n")
    
    # 1. Carregamento dos dados
    print("1. Carregando dados...")
    try:
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
        print(f"   - Train: {train.shape}")
        print(f"   - Test: {test.shape}")
    except FileNotFoundError:
        print("ERRO: Arquivos train.csv e test.csv não encontrados!")
        return
    
    # 2. Limpeza inicial
    print("\n2. Limpeza inicial dos dados...")
    columns_to_remove = ['id', 'category_code']
    for col in columns_to_remove:
        if col in train.columns:
            train = train.drop(columns=[col])
        if col in test.columns:
            test = test.drop(columns=[col])
    
    # 3. Feature Engineering Avançado
    print("\n3. Aplicando feature engineering avançado...")
    X_train = engineer_features_improved(train.drop(columns=['labels']))
    X_test = engineer_features_improved(test)
    y_train = train['labels']
    
    print(f"   - Features criadas: {X_train.shape[1]}")
    print(f"   - Distribuição do target: {y_train.value_counts(normalize=True).round(3).to_dict()}")
    
    # 4. Preprocessamento avançado
    print("\n4. Preprocessamento avançado...")
    X_train_processed, X_test_processed = advanced_preprocessing_pipeline(X_train.copy(), y_train, X_test.copy())
    
    # 5. Split para validação
    print("\n5. Criando split para validação...")
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_processed, y_train, 
        test_size=0.2, 
        stratify=y_train, 
        random_state=42
    )
    
    # 6. Pipeline de ML
    print("\n6. Criando pipeline de ML avançado...")
    pipeline = create_advanced_ml_pipeline()
    
    # 7. Validação cruzada
    print("\n7. Executando validação cruzada...")
    cv_results = cross_validation_analysis(pipeline, X_train_processed, y_train, cv_folds=3)  # Reduzido para eficiência
    
    # 8. Avaliação final
    print("\n8. Avaliação final do modelo...")
    trained_pipeline, optimal_threshold, metrics = comprehensive_model_evaluation(
        pipeline, X_train_split, y_train_split, X_val, y_val
    )
    
    # 9. Análise de importância
    print("\n9. Análise de importância das features...")
    # Pegar nomes das features selecionadas
    try:
        selected_features = X_train_processed.columns[trained_pipeline.named_steps['feature_selection'].support_]
        feature_importance_df = feature_importance_analysis(trained_pipeline, selected_features)
    except:
        print("   - Não foi possível analisar importância das features")
    
    # 10. Predições finais
    print("\n10. Gerando predições finais...")
    test_probabilities = trained_pipeline.predict_proba(X_test_processed)[:, 1]
    test_predictions = (test_probabilities >= optimal_threshold).astype(int)
    
    # 11. Salvar resultados
    print("\n11. Salvando resultados...")
    
    # Submission
    try:
        sample_submission = pd.read_csv('sample_submission.csv')
        sample_submission['labels'] = test_predictions
        sample_submission.to_csv('startup_predictions_improved.csv', index=False)
        print("   - Predições salvas em: startup_predictions_improved.csv")
    except:
        # Criar submission manual
        submission_df = pd.DataFrame({
            'id': range(len(test_predictions)),
            'labels': test_predictions
        })
        submission_df.to_csv('startup_predictions_improved.csv', index=False)
        print("   - Predições salvas em: startup_predictions_improved.csv")
    
    # Salvar probabilidades
    prob_df = pd.DataFrame({
        'id': range(len(test_probabilities)),
        'probability': test_probabilities,
        'prediction': test_predictions
    })
    prob_df.to_csv('startup_probabilities_improved.csv', index=False)
    print("   - Probabilidades salvas em: startup_probabilities_improved.csv")
    
    # Resumo final
    print(f"\n=== RESUMO FINAL ===")
    print(f"Features utilizadas: {X_train_processed.shape[1]}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"Threshold ótimo: {metrics['optimal_threshold']:.3f}")
    print(f"Predições positivas: {test_predictions.sum()}/{len(test_predictions)} ({test_predictions.mean():.1%})")
    
    return trained_pipeline, metrics


if __name__ == "__main__":
    # Executar pipeline principal
    pipeline, metrics = main_pipeline()
    print("\n=== EXECUÇÃO COMPLETA ===")