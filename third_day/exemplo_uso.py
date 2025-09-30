from advanced_binary_classifier import AdvancedBinaryClassifier, create_best_classifier
import pandas as pd

# Carregar dados de exemplo
X_train = pd.read_csv('./train.csv')
y_train = X_train.pop('labels')
novos_dados = pd.read_csv('./test.csv')

# Método 1: Usar a classe diretamente
classifier = AdvancedBinaryClassifier()
classifier.fit(X_train, y_train, quick_mode=False)  # quick_mode=False para melhor performance
predictions = classifier.predict(X_test)

# Método 2: Função de conveniência (mais simples)
classifier = create_best_classifier(X_train, y_train, quick_mode=False)
predictions = classifier.predict(X_test)

# Método 3: A partir de arquivo CSV
classifier = create_best_classifier('./train.csv', target_column='labels')

# Fazer predições (sempre retorna 0 ou 1)
predictions = classifier.predict(novos_dados)
probabilidades = classifier.predict_proba(novos_dados)

# Ver resumo de performance
classifier.print_summary()

# Ver importância das features
importance = classifier.get_feature_importance(top_n=10)
print(importance)
