#Import das bibliotecas utilizadas
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
######################################
#Declaração de constantes para o nome da coluna resposta
RESPONSE ="COVID-19 - PESQUISA DE ANTICORPOS IgG COVID-19 IgG"


'''
Manual do usuário: 
    Basta apenas executar este arquivo que os dados pré-processados
    são separados e os modelos da rede neural são treinados. Resultados
    do balancemento de dados, desempenho dos modelos e a matriz de confusão
    do melhor modelo são salvos no diretório do relatório e demais resultados
    dos treinamentos são mostrados no terminal. 
    Para alterar o diretório de dados pré-processados, basta mudar a variável 'final'.
'''

def main():
    '''Treinamento e teste da rede neural'''
    
    #Registro da tabela com os dados pré-processados utilizando o ID de atendimento como index
    final = pd.read_csv("./PRE/ProcessedData.csv", index_col=0)

    #Seleção dos dados de treinamento e teste do k-fold
    y = final[RESPONSE]
    X = final.drop(columns=[RESPONSE])

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    
    #Distribuição dos dados de treinamento
    plt.hist(y, histtype='bar', color = "skyblue", ec="black", range=(-0.5, 1.5))
    plt.xlabel("Valor da resposta")
    plt.ylabel("Nº de dados")
    plt.savefig('./REL/balanceamento.png')
    plt.show()
    
    ##################################
    #Armazenamento dos indicadores de qualidade do modelo
    foldsScore = []
    foldAccuracy = []
    foldF1 = [] #f score
    FoldPrecision = []
    FoldRecall = [] #cobertura em portugues
    foldCM = []
    models = []
    scalers = []
    
    k = 0

    # treina o modelo para cada fold e guarda a pontuação
    for train_index, test_index in kfold.split(X):
        scaler = MinMaxScaler()
        clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=10000, random_state=3, batch_size=200, 
                            activation='relu', learning_rate='constant', learning_rate_init=0.001)
        
        #Separação dos dados do k-treinamento em teste e treino
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        #Normalização dos dados de treinamento
        X_train_scaled = scaler.fit_transform(X_train)
        #Ajuste dos dados de teste à normalização dos dados de treino
        X_test_scaled = scaler.transform(X_test)
        
        clf.fit(X_train_scaled, y_train)
        score = clf.score(X_test_scaled, y_test)
        prediction = clf.predict(X_test_scaled)
        foldAccuracy.append(accuracy_score(y_test, prediction))
        foldF1.append(f1_score(y_test, prediction))
        FoldPrecision.append(precision_score(y_test, prediction))
        FoldRecall.append(recall_score(y_test, prediction))
        foldsScore.append(score)
        models.append(clf)
        scalers.append(scaler)
        
        print(f"K-treinamento: {k}")
        k += 1
        cm = confusion_matrix(y_test, prediction)
        foldCM.append(cm)
        print(cm)


    foldResults = [FoldRecall, FoldPrecision, foldAccuracy, foldF1]
    foldsTitles = ["recall", "precision", "accuracy", "F1"]

    for i in range(len(foldResults)):
        best = np.max(foldResults[i])
        max_index = np.argmax(foldResults[i])
        print(f"best {foldsTitles[i]}: {best:.4f}, with index {max_index}")
        worstRecall = np.min(foldResults[i])
        min_index = np.argmin(foldResults[i])
        print(f"worst {foldsTitles[i]}: {worstRecall:.4f}, with index {min_index}\n")
        
        score = np.mean(foldResults[i]) #Media das respostas de todos os modelos
        standartDeviation = np.std(foldResults[i])

        print(f"standard deviation: {standartDeviation:.4f}")
        print(f"average {foldsTitles[i]}: {score:.4f}\n")
        print("---------------------------------------------------------------")

    ##################################
    '''
    Análise do desempenho de cada modelo
    '''

    plt.plot([1,2,3,4,5], foldsScore, 'o')
    plt.plot([1,2,3,4,5], foldsScore, c='lightblue')
    plt.xlabel('Índice do modelo treinado')
    plt.ylabel('Acurácia (%)')
    plt.savefig('./REL/desempenho.png')
    plt.show()
    
    ##################################
    '''
    Visualização da matriz de confusão do melhor modelo
    '''
    
    bestModel = np.argmax(foldAccuracy)

    plt.plot([1,2,3,4,5], foldsScore, 'o')
    plt.plot([1,2,3,4,5], foldsScore, c='lightblue')

    disp = ConfusionMatrixDisplay(confusion_matrix=foldCM[bestModel], display_labels=["Não Reagente", "Reagente"])
    disp.plot(cmap="Blues")
    plt.savefig('./REL/matriz_confusao.png')
    plt.show()
    
    

if __name__ == "__main__":
    main()