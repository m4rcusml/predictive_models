
"""
Advanced Binary Classifier
===========================

Sistema modular de classificação binária que testa múltiplos algoritmos
do scikit-learn e cria o melhor modelo possível automaticamente.

Autor: AI Assistant
Data: 2025

Características:
- Testa 8 algoritmos diferentes
- Feature engineering automático
- Seleção de features
- Ensemble dos melhores modelos
- Hyperparameter tuning
- Suporte a dados categóricos e numéricos
- Interface simples e modular

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class AdvancedBinaryClassifier:
    """
    Classe para criar o melhor modelo de classificação binária possível
    usando múltiplos algoritmos do scikit-learn de forma modular.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = None
        self.label_encoders = {}
        self.feature_selector = None
        self.best_model = None
        self.feature_names = None
        self.target_name = None
        self.models_performance = {}

        # Definir modelos base
        self.base_models = {
            'random_forest': RandomForestClassifier(random_state=random_state),
            'gradient_boosting': GradientBoostingClassifier(random_state=random_state),
            'logistic_regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'svm': SVC(random_state=random_state, probability=True),
            'knn': KNeighborsClassifier(),
            'decision_tree': DecisionTreeClassifier(random_state=random_state),
            'naive_bayes': GaussianNB(),
            'ada_boost': AdaBoostClassifier(random_state=random_state)
        }

        # Hiperparâmetros para tuning
        self.param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance']
            }
        }

    def preprocess_data(self, X, y=None, fit_preprocessing=True):
        """
        Preprocessa os dados de entrada
        """
        X_processed = X.copy()

        # Separar colunas numéricas e categóricas
        numeric_columns = X_processed.select_dtypes(include=[np.number]).columns
        categorical_columns = X_processed.select_dtypes(exclude=[np.number]).columns

        if fit_preprocessing:
            # Encodar variáveis categóricas
            self.label_encoders = {}
            for col in categorical_columns:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                self.label_encoders[col] = le

            # Escalar features numéricas
            if len(numeric_columns) > 0:
                self.scaler = RobustScaler()
                X_processed[numeric_columns] = self.scaler.fit_transform(X_processed[numeric_columns])
        else:
            # Aplicar encoders já treinados
            for col in categorical_columns:
                if col in self.label_encoders:
                    # Tratar valores não vistos durante treinamento
                    unique_values = set(self.label_encoders[col].classes_)
                    X_processed[col] = X_processed[col].astype(str).apply(
                        lambda x: x if x in unique_values else 'unknown'
                    )
                    # Adicionar 'unknown' se necessário
                    if 'unknown' not in self.label_encoders[col].classes_:
                        self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, 'unknown')
                    X_processed[col] = self.label_encoders[col].transform(X_processed[col])

            # Aplicar escaler já treinado
            if self.scaler is not None and len(numeric_columns) > 0:
                X_processed[numeric_columns] = self.scaler.transform(X_processed[numeric_columns])

        return X_processed

    def feature_engineering(self, X):
        """
        Cria novas features automaticamente
        """
        X_enhanced = X.copy()
        numeric_columns = X_enhanced.select_dtypes(include=[np.number]).columns

        if len(numeric_columns) > 1:
            # Criar interações entre features numéricas (seleção das mais importantes)
            for i, col1 in enumerate(numeric_columns[:3]):  # Limitar para evitar explosão de features
                for col2 in numeric_columns[i+1:4]:
                    X_enhanced[f'{col1}_x_{col2}'] = X_enhanced[col1] * X_enhanced[col2]
                    X_enhanced[f'{col1}_div_{col2}'] = X_enhanced[col1] / (X_enhanced[col2] + 1e-8)

            # Criar estatísticas agregadas
            X_enhanced['numeric_sum'] = X_enhanced[numeric_columns].sum(axis=1)
            X_enhanced['numeric_mean'] = X_enhanced[numeric_columns].mean(axis=1)
            X_enhanced['numeric_std'] = X_enhanced[numeric_columns].std(axis=1)

        return X_enhanced

    def select_features(self, X, y, method='rfe', k=20):
        """
        Seleciona as melhores features
        """
        if method == 'selectk':
            self.feature_selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
        elif method == 'rfe':
            # Usar Random Forest para RFE (mais eficiente)
            estimator = RandomForestClassifier(n_estimators=50, random_state=self.random_state)
            self.feature_selector = RFE(estimator, n_features_to_select=min(k, X.shape[1]))

        X_selected = self.feature_selector.fit_transform(X, y)
        return X_selected

    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test, quick_mode=False):
        """
        Treina e avalia todos os modelos
        """
        print("Treinando e avaliando modelos...")

        for name, model in self.base_models.items():
            print(f"\nTreinando {name}...")

            if not quick_mode and name in self.param_grids:
                # Hyperparameter tuning
                search = RandomizedSearchCV(
                    model, 
                    self.param_grids[name], 
                    n_iter=10, 
                    cv=3, 
                    scoring='roc_auc',
                    random_state=self.random_state,
                    n_jobs=-1
                )
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
            else:
                # Treinar modelo com parâmetros padrão
                best_model = model
                best_model.fit(X_train, y_train)

            # Avaliar modelo
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else y_pred

            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            self.models_performance[name] = {
                'model': best_model,
                'accuracy': accuracy,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'cv_score': cross_val_score(best_model, X_train, y_train, cv=3, scoring='roc_auc').mean()
            }

            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  ROC-AUC: {roc_auc:.4f}")

    def create_ensemble(self, X_train, y_train, top_n=3):
        """
        Cria um ensemble com os melhores modelos
        """
        print(f"\nCriando ensemble com os top {top_n} modelos...")

        # Ordenar modelos por performance
        sorted_models = sorted(
            self.models_performance.items(), 
            key=lambda x: x[1]['roc_auc'], 
            reverse=True
        )

        # Selecionar os melhores modelos
        top_models = [(name, data['model']) for name, data in sorted_models[:top_n]]

        # Criar ensemble
        ensemble = VotingClassifier(
            estimators=top_models,
            voting='soft'  # Usar probabilidades
        )

        ensemble.fit(X_train, y_train)
        return ensemble

    def fit(self, X, y, test_size=0.2, feature_engineering=True, feature_selection=True, 
            ensemble=True, quick_mode=False):
        """
        Treina o modelo completo
        """
        print("Iniciando treinamento do modelo avançado...")

        # Armazenar nomes das colunas
        self.feature_names = X.columns.tolist() if hasattr(X, 'columns') else None
        self.target_name = y.name if hasattr(y, 'name') else 'target'

        # Converter para DataFrame se necessário
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # Preprocessamento
        print("Preprocessando dados...")
        X_processed = self.preprocess_data(X, fit_preprocessing=True)

        # Feature Engineering
        if feature_engineering:
            print("Criando novas features...")
            X_processed = self.feature_engineering(X_processed)

        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Seleção de features
        if feature_selection:
            print("Selecionando features...")
            X_train_selected = self.select_features(X_train, y_train)
            X_test_selected = self.feature_selector.transform(X_test)
        else:
            X_train_selected = X_train
            X_test_selected = X_test

        # Treinar modelos
        self.train_and_evaluate_models(X_train_selected, X_test_selected, y_train, y_test, quick_mode)

        # Criar ensemble
        if ensemble:
            self.best_model = self.create_ensemble(X_train_selected, y_train)

            # Avaliar ensemble
            y_pred_ensemble = self.best_model.predict(X_test_selected)
            y_pred_proba_ensemble = self.best_model.predict_proba(X_test_selected)[:, 1]

            ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
            ensemble_f1 = f1_score(y_test, y_pred_ensemble)
            ensemble_roc_auc = roc_auc_score(y_test, y_pred_proba_ensemble)

            print(f"\nPerformance do Ensemble:")
            print(f"  Accuracy: {ensemble_accuracy:.4f}")
            print(f"  F1-Score: {ensemble_f1:.4f}")
            print(f"  ROC-AUC: {ensemble_roc_auc:.4f}")
        else:
            # Usar o melhor modelo individual
            best_name = max(self.models_performance.items(), key=lambda x: x[1]['roc_auc'])[0]
            self.best_model = self.models_performance[best_name]['model']

        return self

    def predict(self, X):
        """
        Faz predições (retorna 0 ou 1)
        """
        # Preprocessar dados
        X_processed = self.preprocess_data(X, fit_preprocessing=False)

        # Feature engineering
        X_processed = self.feature_engineering(X_processed)

        # Seleção de features
        if self.feature_selector is not None:
            X_processed = self.feature_selector.transform(X_processed)

        return self.best_model.predict(X_processed)

    def predict_proba(self, X):
        """
        Retorna probabilidades das predições
        """
        # Preprocessar dados
        X_processed = self.preprocess_data(X, fit_preprocessing=False)

        # Feature engineering
        X_processed = self.feature_engineering(X_processed)

        # Seleção de features
        if self.feature_selector is not None:
            X_processed = self.feature_selector.transform(X_processed)

        return self.best_model.predict_proba(X_processed)

    def get_feature_importance(self, top_n=10):
        """
        Retorna a importância das features (quando possível)
        """
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'estimators_'):
            # Para ensemble, tentar obter importâncias médias
            try:
                importances = np.mean([
                    est.feature_importances_ if hasattr(est, 'feature_importances_') 
                    else np.abs(est.coef_[0]) if hasattr(est, 'coef_') 
                    else np.zeros(len(self.feature_names))
                    for name, est in self.best_model.estimators_
                ], axis=0)
            except:
                return "Importância das features não disponível para este ensemble"
        else:
            return "Importância das features não disponível para este modelo"

        # Criar DataFrame com importâncias
        feature_names = self.feature_names if self.feature_names else [f'feature_{i}' for i in range(len(importances))]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)

        return importance_df

    def print_summary(self):
        """
        Imprime resumo dos resultados
        """
        print("\n" + "="*50)
        print("RESUMO DOS MODELOS")
        print("="*50)

        for name, metrics in sorted(self.models_performance.items(), key=lambda x: x[1]['roc_auc'], reverse=True):
            print(f"\n{name.upper()}:")
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  CV Score: {metrics['cv_score']:.4f}")


# FUNÇÕES UTILITÁRIAS

def create_best_classifier(X, y, target_column=None, quick_mode=False, verbose=True):
    """
    Função de conveniência para criar rapidamente o melhor classificador

    Args:
        X: Features (DataFrame, array, ou arquivo CSV)
        y: Target (Series, array, ou nome da coluna se X contém target)
        target_column: Nome da coluna target (se X contém target)
        quick_mode: Se True, usa parâmetros padrão (mais rápido)
        verbose: Se True, mostra progresso

    Returns:
        modelo treinado
    """

    # Carregar dados se for string (caminho do arquivo)
    if isinstance(X, str):
        if X.endswith('.csv'):
            df = pd.read_csv(X)
            if target_column:
                y = df[target_column]
                X = df.drop(columns=[target_column])
            else:
                raise ValueError("Especifique target_column para arquivos CSV")

    # Converter para DataFrame/Series se necessário
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(y, (pd.Series, np.ndarray)):
        y = pd.Series(y)

    # Criar e treinar classificador
    classifier = AdvancedBinaryClassifier()
    classifier.fit(X, y, quick_mode=quick_mode)

    if verbose:
        classifier.print_summary()

    return classifier


def evaluate_model(model, X_test, y_test):
    """
    Avalia um modelo em novos dados
    """
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print("AVALIAÇÃO EM DADOS DE TESTE:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'predictions': predictions,
        'probabilities': probabilities
    }


# EXEMPLO DE USO
if __name__ == "__main__":
    # Exemplo básico
    from sklearn.datasets import make_classification

    # Criar dados de exemplo
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_df = pd.Series(y, name='target')

    # Usar a classe
    classifier = AdvancedBinaryClassifier()
    classifier.fit(X_df, y_df, quick_mode=True)

    # Fazer predições
    predictions = classifier.predict(X_df[:5])
    probabilities = classifier.predict_proba(X_df[:5])

    print(f"\nPredições: {predictions}")
    print(f"Probabilidades: {probabilities[:, 1]}")

    # Ou usar a função de conveniência
    # classifier = create_best_classifier(X_df, y_df, quick_mode=True)
