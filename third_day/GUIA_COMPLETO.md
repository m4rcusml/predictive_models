
# Advanced Binary Classifier - Guia Completo

## Visão Geral

O **Advanced Binary Classifier** é um sistema modular de machine learning que automaticamente testa múltiplos algoritmos do scikit-learn e cria o melhor modelo de classificação binária possível para seus dados.

### Características Principais

✅ **8 Algoritmos Diferentes**: Random Forest, Gradient Boosting, Logistic Regression, SVM, KNN, Decision Tree, Naive Bayes, AdaBoost  
✅ **Feature Engineering Automático**: Cria novas features baseadas em interações e estatísticas  
✅ **Seleção de Features**: Remove features irrelevantes automaticamente  
✅ **Ensemble Learning**: Combina os melhores modelos  
✅ **Hyperparameter Tuning**: Otimiza parâmetros automaticamente  
✅ **Suporte a Dados Categóricos**: Encoding automático de variáveis categóricas  
✅ **Interface Simples**: Fácil de usar com qualquer dataset  
✅ **Análise Completa**: Relatórios detalhados de performance  

## Instalação

```python
# Dependências necessárias
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Uso Básico

### 1. Importar e Usar

```python
from advanced_binary_classifier import AdvancedBinaryClassifier
import pandas as pd

# Carregar seus dados
X = pd.DataFrame(...)  # Features
y = pd.Series(...)     # Target (0 ou 1)

# Criar e treinar modelo
classifier = AdvancedBinaryClassifier()
classifier.fit(X, y)

# Fazer predições
predictions = classifier.predict(X_test)  # Retorna 0 ou 1
probabilities = classifier.predict_proba(X_test)  # Retorna probabilidades
```

### 2. Função de Conveniência

```python
from advanced_binary_classifier import create_best_classifier

# Forma mais simples
classifier = create_best_classifier(X, y)
predictions = classifier.predict(X_test)
```

### 3. A partir de Arquivo CSV

```python
# Se seus dados estão em CSV
classifier = create_best_classifier('dados.csv', target_column='target')
```

## Configurações Avançadas

### Controle de Feature Engineering

```python
classifier.fit(
    X, y,
    feature_engineering=True,   # Criar novas features (padrão: True)
    feature_selection=True,     # Selecionar melhores features (padrão: True)
    ensemble=True,              # Usar ensemble (padrão: True)
    quick_mode=False            # Hyperparameter tuning completo (padrão: False)
)
```

### Modo Rápido vs Completo

```python
# Modo rápido (mais rápido, menos otimizado)
classifier.fit(X, y, quick_mode=True)

# Modo completo (mais lento, melhor performance)
classifier.fit(X, y, quick_mode=False)
```

## Análise de Resultados

### 1. Resumo dos Modelos

```python
classifier.print_summary()
# Mostra performance de todos os modelos testados
```

### 2. Importância das Features

```python
importance = classifier.get_feature_importance(top_n=10)
print(importance)
```

### 3. Avaliação Detalhada

```python
from advanced_binary_classifier import evaluate_model

results = evaluate_model(classifier, X_test, y_test)
print(f"Accuracy: {results['accuracy']}")
print(f"F1-Score: {results['f1_score']}")
print(f"ROC-AUC: {results['roc_auc']}")
```

## Otimização de Threshold

```python
# Otimizar threshold para maximizar F1-score
def optimize_threshold(classifier, X_val, y_val):
    from sklearn.metrics import precision_recall_curve, f1_score

    y_probs = classifier.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    return best_threshold

# Usar threshold otimizado
threshold = optimize_threshold(classifier, X_val, y_val)
predictions = (classifier.predict_proba(X_test)[:, 1] >= threshold).astype(int)
```

## Exemplos Práticos

### 1. Dataset de E-commerce

```python
# Exemplo: Prever se cliente vai comprar
import pandas as pd

# Dados do cliente
df = pd.read_csv('clientes.csv')
X = df[['idade', 'renda', 'visitas_site', 'tempo_sessao', 'genero', 'regiao']]
y = df['comprou']  # 0 = não comprou, 1 = comprou

# Treinar modelo
classifier = AdvancedBinaryClassifier()
classifier.fit(X, y, quick_mode=False)

# Ver performance
classifier.print_summary()

# Predições para novos clientes
novos_clientes = pd.DataFrame({...})
predicoes = classifier.predict(novos_clientes)
probabilidades = classifier.predict_proba(novos_clientes)[:, 1]
```

### 2. Dataset de Saúde

```python
# Exemplo: Diagnóstico médico
df = pd.read_csv('dados_medicos.csv')
X = df.drop('diagnostico', axis=1)  # Features médicas
y = df['diagnostico']  # 0 = negativo, 1 = positivo

# Treinar com foco em recall (detectar todos os casos positivos)
classifier = AdvancedBinaryClassifier()
classifier.fit(X, y)

# Análise detalhada
from sklearn.metrics import classification_report
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))
```

### 3. Dataset Financeiro

```python
# Exemplo: Detecção de fraude
df = pd.read_csv('transacoes.csv')
X = df[['valor', 'horario', 'tipo_cartao', 'merchant', 'localizacao']]
y = df['fraude']  # 0 = legítimo, 1 = fraude

# Treinar modelo
classifier = AdvancedBinaryClassifier()
classifier.fit(X, y)

# Monitorar novas transações
novas_transacoes = pd.DataFrame({...})
risco_fraude = classifier.predict_proba(novas_transacoes)[:, 1]

# Alertar se probabilidade > 0.8
alertas = risco_fraude > 0.8
```

## Tratamento de Dados

### Dados Categóricos
O sistema automaticamente detecta e codifica variáveis categóricas usando Label Encoding.

### Dados Numéricos
Features numéricas são automaticamente escaladas usando Robust Scaler (resistente a outliers).

### Dados Faltantes
Trate dados faltantes antes de usar o classificador:

```python
# Exemplo de tratamento
df = df.fillna(df.median())  # Para numéricos
df = df.fillna(df.mode().iloc[0])  # Para categóricos
```

## Métricas de Performance

O sistema avalia modelos usando:

- **ROC-AUC**: Área sob a curva ROC (principal métrica)
- **F1-Score**: Harmônica entre precision e recall
- **Accuracy**: Porcentagem de predições corretas
- **Cross-Validation Score**: Validação cruzada

## Dicas de Performance

1. **Use quick_mode=False** para máxima performance
2. **Mais dados = melhor modelo** (mínimo 1000 amostras recomendado)
3. **Balance o dataset** se possível
4. **Trate outliers** antes do treinamento
5. **Valide em dados não vistos** sempre

## Troubleshooting

### Problema: Performance baixa
- Verifique qualidade dos dados
- Aumente o tamanho do dataset
- Trate outliers e dados faltantes
- Use feature_engineering=True

### Problema: Muito lento
- Use quick_mode=True para testes
- Reduza o número de features
- Use feature_selection=True

### Problema: Overfitting
- Aumente o conjunto de teste
- Use validação cruzada
- Reduza complexidade do modelo

## Integração com Produção

```python
# Salvar modelo treinado
import pickle

# Treinar modelo
classifier = AdvancedBinaryClassifier()
classifier.fit(X_train, y_train)

# Salvar
with open('modelo_producao.pkl', 'wb') as f:
    pickle.dump(classifier, f)

# Carregar em produção
with open('modelo_producao.pkl', 'rb') as f:
    modelo = pickle.load(f)

# Usar
predictions = modelo.predict(novos_dados)
```

## Comparação com Outros Métodos

| Método | Automação | Performance | Flexibilidade |
|--------|-----------|-------------|---------------|
| Advanced Binary Classifier | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Modelo único (ex: Random Forest) | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| AutoML (ex: AutoSklearn) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| Desenvolvimento manual | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## FAQ

**P: Funciona com datasets pequenos?**  
R: Sim, mas performance é melhor com 1000+ amostras.

**P: Posso usar com mais de 2 classes?**  
R: Não, é específico para classificação binária (0/1).

**P: Como escolher entre quick_mode True/False?**  
R: Use True para testes rápidos, False para produção.

**P: O modelo funciona com dados temporais?**  
R: Sim, mas trate features temporais adequadamente antes.

**P: Posso personalizar os algoritmos?**  
R: Sim, modifique o código fonte da classe.

## Suporte

Para dúvidas ou problemas:
1. Verifique este guia primeiro
2. Teste com dados sintéticos
3. Verifique logs de erro
4. Revise pré-processamento dos dados

---

**Versão:** 1.0  
**Compatibilidade:** Python 3.7+, scikit-learn 0.24+
