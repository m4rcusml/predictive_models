
"""
EXEMPLO AVANÇADO - ADVANCED BINARY CLASSIFIER
==============================================

Este exemplo mostra como usar o classificador com otimizações avançadas
e análise detalhada dos resultados.

"""

from advanced_binary_classifier import AdvancedBinaryClassifier, create_best_classifier, evaluate_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def create_advanced_model_with_analysis(X, y, test_size=0.2):
    """
    Cria modelo avançado com análise completa dos resultados
    """

    # Dividir dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print("CONFIGURAÇÃO DO DATASET:")
    print(f"- Treino: {X_train.shape[0]} amostras")
    print(f"- Teste: {X_test.shape[0]} amostras")
    print(f"- Features: {X_train.shape[1]}")
    print(f"- Distribuição treino: {dict(y_train.value_counts())}")
    print(f"- Distribuição teste: {dict(y_test.value_counts())}")

    # Criar modelo com configurações otimizadas
    classifier = AdvancedBinaryClassifier(random_state=42)

    # Treinar com todas as otimizações
    classifier.fit(
        X_train, 
        y_train, 
        test_size=0.2,          # Validação interna
        feature_engineering=True, # Criar novas features
        feature_selection=True,   # Selecionar melhores features
        ensemble=True,           # Usar ensemble
        quick_mode=False         # Fazer hyperparameter tuning
    )

    # Avaliar no conjunto de teste
    print("\n" + "="*50)
    print("AVALIAÇÃO NO CONJUNTO DE TESTE")
    print("="*50)

    results = evaluate_model(classifier, X_test, y_test)

    # Relatório de classificação detalhado
    y_pred = results['predictions']
    print("\nRELATÓRIO DE CLASSIFICAÇÃO:")
    print(classification_report(y_test, y_pred))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    print("\nMATRIZ DE CONFUSÃO:")
    print(cm)

    # Análise de probabilidades
    probs = results['probabilities']
    print(f"\nANÁLISE DE PROBABILIDADES:")
    print(f"- Probabilidade média para classe 1: {probs.mean():.3f}")
    print(f"- Desvio padrão: {probs.std():.3f}")
    print(f"- Min: {probs.min():.3f}, Max: {probs.max():.3f}")

    # Importância das features
    importance = classifier.get_feature_importance(top_n=15)
    if isinstance(importance, pd.DataFrame):
        print("\nTOP 15 FEATURES MAIS IMPORTANTES:")
        for idx, row in importance.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")

    return classifier, results

def optimize_threshold(classifier, X_val, y_val):
    """
    Otimiza o threshold de decisão para maximizar F1-score
    """
    from sklearn.metrics import precision_recall_curve

    # Obter probabilidades
    y_probs = classifier.predict_proba(X_val)[:, 1]

    # Calcular precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_val, y_probs)

    # Calcular F1-score para cada threshold
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)  # Tratar divisão por zero

    # Encontrar melhor threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"\nOTIMIZAÇÃO DE THRESHOLD:")
    print(f"- Melhor threshold: {best_threshold:.3f}")
    print(f"- F1-score otimizado: {best_f1:.4f}")
    print(f"- Precision: {precision[best_idx]:.4f}")
    print(f"- Recall: {recall[best_idx]:.4f}")

    return best_threshold

def predict_with_custom_threshold(classifier, X, threshold=0.5):
    """
    Faz predições com threshold customizado
    """
    probs = classifier.predict_proba(X)[:, 1]
    return (probs >= threshold).astype(int)

# EXEMPLO PRINCIPAL
if __name__ == "__main__":

    print("="*60)
    print("EXEMPLO AVANÇADO - ADVANCED BINARY CLASSIFIER")
    print("="*60)

    # Criar dataset mais desafiador
    X, y = make_classification(
        n_samples=2000,
        n_features=25,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=2,
        class_sep=0.8,  # Tornar mais desafiador
        random_state=42
    )

    # Converter para DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_df = pd.Series(y, name='target')

    # Criar modelo avançado
    classifier, results = create_advanced_model_with_analysis(X_df, y_df)

    # Otimizar threshold em dados de validação
    X_train, X_val, y_train, y_val = train_test_split(X_df, y_df, test_size=0.2, random_state=42)
    best_threshold = optimize_threshold(classifier, X_val, y_val)

    # Comparar predições com threshold padrão vs otimizado
    print("\n" + "="*50)
    print("COMPARAÇÃO DE THRESHOLDS")
    print("="*50)

    # Threshold padrão (0.5)
    pred_default = classifier.predict(X_val)
    f1_default = f1_score(y_val, pred_default)

    # Threshold otimizado
    pred_optimized = predict_with_custom_threshold(classifier, X_val, best_threshold)
    f1_optimized = f1_score(y_val, pred_optimized)

    print(f"F1-Score com threshold 0.5: {f1_default:.4f}")
    print(f"F1-Score com threshold otimizado: {f1_optimized:.4f}")
    print(f"Melhoria: {f1_optimized - f1_default:.4f}")

    # Exemplo de uso em produção
    print("\n" + "="*50)
    print("EXEMPLO DE USO EM PRODUÇÃO")
    print("="*50)

    # Simular novos dados
    new_data = X_df.sample(10, random_state=123).reset_index(drop=True)

    print("Predições para novos dados:")
    for i in range(len(new_data)):
        prob = classifier.predict_proba(new_data.iloc[[i]])[:, 1][0]
        pred_default = classifier.predict(new_data.iloc[[i]])[0]
        pred_optimized = predict_with_custom_threshold(classifier, new_data.iloc[[i]], best_threshold)[0]

        print(f"Amostra {i+1}:")
        print(f"  - Probabilidade: {prob:.3f}")
        print(f"  - Predição (threshold=0.5): {pred_default}")
        print(f"  - Predição (threshold={best_threshold:.3f}): {pred_optimized}")

    print("\n" + "="*60)
    print("MODELO TREINADO E OTIMIZADO COM SUCESSO!")
    print("="*60)
    print("\nO modelo está pronto para uso em produção!")
    print("Use classifier.predict() para predições ou")
    print("predict_with_custom_threshold() para threshold customizado")
