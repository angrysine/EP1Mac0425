#Import das bibliotecas utilizadas
import numpy as np
import pandas as pd
from unidecode import unidecode
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
######################################
#Declaração de constantes para nomes das colunas importantes
RESPONSE = "COVID-19 - PESQUISA DE ANTICORPOS IgG COVID-19 IgG"

#Colunas com menos volumes de dados que a resposta contendo exames de covid-19
COVIDRELATED1 = "TESTE RÁPIDO PARA COVID-19 DE PONTA DE DEDO TESTE RÁPIDO PARA COVID-19 DE PONTA DE DEDO"
COVIDRELATED2 = "COVID TESTE LÍQUOR COVID TESTE LÍQUOR"
COVIDRELATED3 = "Teste Rápido para SARS-CoV-2- Pesquisa de anticorpos IgG e IgM (Sorologia para COVID-19) Teste Rápido para SARS-CoV-2- Pesquisa de anticor"

#Declaração de constantes de palavras com viés positivo ou negativo no ambiente médico
POSITIVE_WORDS = ["pos", "numeroso", "positivo", "grande", "detectado", "detectavel", "identificado", "reagente", "sim", 
                    "presente", "presenca", "maximo", "observado", "s", "numerosas", "numerosos", "normais", 
                    "intenso", "intensa", "positivos", "positivas", "reagentes", "+", "ligeiramente", 
                    "moderado", "moderados", "acima", "conservada", "alguns", "algumas", "++", "+++", "pequeno",
                    "raro", "rara", "realizado", "discreta", "discreto", "forte", "moderada", "moderadas", "tracos", "presença",
                    "levemente", "fortemente"]

NEGATIVE_WORDS = ["neg", "nao", "negativo", "isolados", "inadequada", "zero", 
                "invalido", "invalido", "ausencia", "ausente", "ausentes", "ausencia", "failed", "indetectavel",
                "minima", "r", "negativos", "negativas", "n", "w", "*", "normal",
                "zero", "rarissimos", "abaixo", "diminuida", "sem", "norm", "comum", "isolados", "isoladas",
                "isolado"]

######################################
#Definição de funções auxiliares
def try_float(x):
    '''
    (str) -> float or NaN
    Esta função recebe uma string e tenta converter
    ela para float, verificando se ela possui marcações
    como frações ou desigualdades, igualando a números
    de ponto flutuante aproximados
    '''
    try:
        if "/" in x:
            return frac_to_float(x)
        elif ">" in x:
            return float(x.split(">")[-1])
        elif "<" in x:
            return float(x.split("<")[-1])
            
        return float(x)
    except (ValueError, TypeError):
        return np.nan
        
def frac_to_float(frac_str):
    '''
    (str) -> float
    Esta função recebe uma string representada
    como fração e converte ela para um float
    '''
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac

######################################

def main():
    ''' Função para o pré-processamento de dados para o treinamento e teste da rede neural '''
    
    ##################################
    '''Aqui são importadas as tabelas de pacientes e exames além
        da criação de uma tabela pivot em função do id de atendimento,
        junto com a filtragem e aumento no volume de dados.'''
    
    pacientes = pd.read_csv(r'./Data/HC_PACIENTES_1.csv', sep='|')

    df = pd.read_csv(r'./Data/HC_EXAMES_1.csv', sep='|')
    df = pd.merge(df, pacientes) # Junta as tabelas de exames e pacientes pelo ID do paciente

    df.insert(1, "EXAME", df['DE_EXAME']+" "+df['DE_ANALITO'])

    out = df.pivot_table(index=['ID_aTENDIMENTO', 'IC_SEXO', 'AA_NASCIMENTO'], columns='EXAME', values='DE_RESULTADO', aggfunc='first')
    out = out.reset_index(level=['IC_SEXO', 'AA_NASCIMENTO'])

    del out['COVID-19 - PESQUISA DE ANTICORPOS IgG Índice:']

    le = LabelEncoder()

    #Aumenta o volume de dados checando se há outro teste relacionado à covid e se deu positivo, então o Igg deve ser negativo
    out[RESPONSE].mask(out[RESPONSE].isna() & (out[COVIDRELATED1].notna() | out[COVIDRELATED2].notna() | out[COVIDRELATED3].notna()), "Não reagente", inplace=True)

    #Tirar colunas sem resposta 
    out.dropna(subset=[RESPONSE], inplace=True)
    #0 negativo, 1 positivo, removemos classe indeterminado
    out = out[out[RESPONSE].isin(["Não reagente", "Reagente"])]
    
    
    ##################################
    '''
    Aqui a qualidade dos dados é melhorada, eles são convertidos em binários (0 ou 1)
    caso sejam identificados como um campo não numérico e os valores nulos são
    preenchidos como 0 caso o campo seja numérico ou 0.5, caso seja binário.
    '''

    mapping = {}
    def replace_enconde(x):
        '''
        (str -> int or NaN)
        Identifica se na string de entrada há alguma palavra considerada um resultado
        negativo ou positivo no contexto médico e retorna um valor binário (0 ou 1)
        dependendo do significado. Caso não ache nada assim, vê se a string de entrada
        está no conjunto das 2 palavras mais utilizadas da coluna da tabela, novamente
        retornando um valor binário.
        '''
        if x is np.nan: return np.nan
        
        if any(text in NEGATIVE_WORDS for text in unidecode(x.lower()).split(" ")):
            return 0
        elif any(text in POSITIVE_WORDS for text in unidecode(x.lower()).split(" ")):
            return 1
        else:
            try:
                return mapping[x]
            except:
                return np.nan

    final = pd.DataFrame(out) #Planilha final

    #Função para converter os valores de uma coluna para float
    def convertNumber(column):
        converted_column = final[column].map(try_float)
        final[column] = converted_column

    for column in out.columns[2:]:
        #Dados de data, nomes de pessoas (colunas 'conferido' e 'realizado') e horários (colunas 'detecção:') não tem valor e são descartados
        if any(text in ["data", "conferido", "realizado", "detecção:"] for text in column.lower().split(" ")):
            del final[column]
            continue
        
        countings = out[column].value_counts().index.tolist()
        
        encodeList = []
        convertido = False
        
        #Aqui os 2 valores com maior contagem da coluna são armazenados e é visto de é um valor numérico ou binário
        for i in range(0, min(2, len(countings))):
            try:
                float(countings[i])
                # Aplica em cada valor da coluna
                convertNumber(column)
                convertido = True
                break
            except:
                encodeList.append(countings[i])
        
        #Se ainda não foi convertido para numérico, converte para binário
        if not convertido:
            ids = le.fit_transform(encodeList)
            mapping = dict(zip(le.classes_, range(len(le.classes_))))
            converted_column = final[column].map(replace_enconde)
            final[column] = converted_column

    #Diminuição de colunas com poucos dados
    for column in final.columns:
        if final[column].count() < 0.1 * len(final.index):
            del final[column]
            continue

    #Preenche todos os valores NaN com o valor pedido
    for col in final.columns[2:]:
        n_unique = final[col].nunique(dropna=True)
        
        fill_value = 0.5 if n_unique == 2 else 0
        final[col] = final[col].fillna(fill_value)
        
    #Atualização dos sexos para dados binários
    final['IC_SEXO'] = le.fit_transform(final['IC_SEXO'])

    #Atualização das idades
    def calculate_age(born):
        #Checando o valor nulo padrão de datas
        try:
            return 2020 - int(born)
        except: 
            return np.nan

    final['AA_NASCIMENTO'] = final['AA_NASCIMENTO'].map(calculate_age)
    #Dados nulos são preenchidos com a mediana das idades
    final['AA_NASCIMENTO'].fillna(final['AA_NASCIMENTO'].median(), inplace=True)

    final.to_csv("./PRE/ProcessedData.csv")


if __name__ == "__main__":
    main()