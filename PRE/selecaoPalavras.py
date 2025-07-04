'''Algoritmo para a visualização e seleção das palavras com aspectos positivos
    ou negativos da planilha'''
    
import pandas as pd

def has_digits(line):
    return any(char.isdigit() for char in line)

def main():
    df = pd.read_csv(r'./Data/HC_EXAMES_1.csv', sep='|')
    df.insert(1, "EXAME", df['DE_EXAME']+" "+df['DE_ANALITO'])
    out = df.pivot_table(index=['ID_aTENDIMENTO'], columns='EXAME', values='DE_RESULTADO', aggfunc='first')
    
    valuesTable = []

    for column in out.columns:
        uniqueValues = out[column].unique()
        notNullVal = uniqueValues[~pd.isnull(uniqueValues)]
        values = notNullVal.transpose()

        valuesTable.append(values)

    stringValues = []

    for values in valuesTable:
        for value in values:
            try:
                float(value)
            except:
                if not has_digits(value):
                    stringValues.append(value)

    stringValues = pd.DataFrame(stringValues)[0].unique()
    
    for item in stringValues:
        print(item)
    
if __name__=="__main__":
    main()