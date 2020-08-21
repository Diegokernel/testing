# -*- coding: utf-8 -*-
"""
File to process diabetes prediction
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import collections
from scipy import stats
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.preprocessing import StandardScaler


home = str(Path.home())
df_med = pd.read_excel(os.path.join(home, "Downloads/CAC RISARALDA_JUNIO2020.xlsx"),  sep=r"\s*,\s*")
df_med.rename(columns=lambda x: x.strip(), inplace=True)
del df_med['Observacion']
print(df_med.info())


def barplot1(column, title):
    carrier_count = column.value_counts()
    sns.set(style="darkgrid")
    sns.barplot(carrier_count.index, carrier_count.values, alpha=0.9)
    
    plt.title(title)
    plt.ylabel('Número de ocurrencias', fontsize=12)
    plt.xlabel('Riesgo', fontsize=12)
    plt.show()
    
def piechart1(column):
    labels = column.astype('category').cat.categories.tolist()
    counts = column.value_counts()
    sizes = [counts[var_cat] for var_cat in labels]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
    ax1.axis('equal')
    plt.show()

df_med['HTA y DM'] = df_med['Diagnóstico HTA'].str.cat(df_med['Diagnóstico DM'],sep=" ")
print(df_med['HTA y DM'].isnull().sum())
htadm = df_med['HTA y DM']
replace_map_med_htadm = {'HTA y DM': {'SI NO': 5, 'NO SI': 3, 'SI SI': 7, 'NO NO': 0}}
df_med.replace(replace_map_med_htadm, inplace=True)
title = 'Distribución de Frecuencias HTA y DM'

barplot1(htadm, title)
piechart1(htadm)
#Asimetria negativa
df_med['HTA y DM'].skew()
#leptocurtica
df_med['HTA y DM'].kurtosis()
df_med['HTA y DM'].describe()


df_med['HTA'] = df_med['Diagnóstico HTA'].eq('SI').mul(1)
df_med['DM'] = df_med['Diagnóstico DM'].eq('SI').mul(1)


print(df_med["Hemoglobina glicosilada"].isnull().sum())
hba1c = []
Hemoglobina = []

a = pd.to_numeric(df_med["Hemoglobina glicosilada"], errors='coerce')
stats.pointbiserialr(a.fillna(0),df_med['HTA'])
stats.pointbiserialr(a.fillna(0),df_med['DM'])

for val in a:
    if val < 4  or val > 20:
        hba1c.append(None)
    elif val < 7.5:
        hba1c.append(1)
    elif val >= 7.5 and val <= 9:
        hba1c.append(3)
    elif val > 9:
        hba1c.append(7)
    else:
        hba1c.append(0)
           
df_med['HbA1c'] = hba1c
barplot1(df_med['HbA1c'], "Distribución de Fecuencias HbA1c")
piechart1(df_med['HbA1c']) 
#Asimetria positiva
df_med['HbA1c'].skew()
#leptocúrtica
df_med['HbA1c'].kurtosis()
df_med['HbA1c'].describe()
stats.pointbiserialr(df_med['HbA1c'].fillna(0),df_med['HTA'])
stats.pointbiserialr(df_med['HbA1c'].fillna(0),df_med['DM'])


b = pd.to_numeric(df_med['Presión arterial sistólica'], errors='coerce')
c = pd.to_numeric(df_med['Presión arterial diastólica'], errors='coerce')

press = []
for sis, dias in zip(b, c):
    if sis < 90 or sis > 250:
        press.append(None)
    elif dias < 60 or dias > 140:
        press.append(None)
    else:
        press.append(sis/dias)

df_med['Press'] = press
stats.pointbiserialr(df_med['Press'].fillna(0),df_med['HTA'])
stats.pointbiserialr(df_med['Press'].fillna(0),df_med['DM'])

Control_HTA = []
for edad, press in zip(pd.to_numeric(df_med['Edad'], errors='coerce'), df_med['Press']): 
    if press == None:
        Control_HTA.append(None)
    elif edad < 60 and press < 1.55555555556:
        Control_HTA.append(1)
    elif edad < 60 and press >= 1.55555555556:
        Control_HTA.append(3)
    elif edad > 60 and press < 1.55555555556:
        Control_HTA.append(1)
    elif edad > 60 and press >= 1.55555555556:
        Control_HTA.append(5)
    else:
        Control_HTA.append(0)

len(Control_HTA)
df_med['Control HTA'] = Control_HTA
barplot1(df_med['Control HTA'], "Distribución de Fecuencias HTA")
piechart1(df_med['Control HTA'])

#Asimetria negativa
df_med['Control HTA'].skew()
#platicúrtica
df_med['Control HTA'].kurtosis()
df_med['Control HTA'].describe()
stats.pointbiserialr(df_med['Control HTA'].fillna(0),df_med['HTA'])
stats.pointbiserialr(df_med['Control HTA'].fillna(0),df_med['DM'])

LDL = []
g = pd.to_numeric(df_med['Colesterol LDL'], errors='coerce')
#correlacion positiva
g.cov(df_med['HTA'])
#correlacion negativa
g.cov(df_med['DM'])


for colesterol in pd.to_numeric(df_med['Colesterol LDL'], errors='coerce'):
    if colesterol < 70 or colesterol > 250:
        LDL.append(None)
    elif colesterol <= 100:
        LDL.append(1)
    elif colesterol > 100 and colesterol < 190:
        LDL.append(5)
    elif colesterol >= 190:
        LDL.append(7)
    else:
        LDL.append(0)
        
df_med['LDL L'] = LDL
barplot1(df_med['LDL L'], "Distribución de Fecuencias LDL")
piechart1(df_med['LDL L'])
#Asimetria negativa
df_med['LDL L'].skew()
#platicúrtica
df_med['LDL L'].kurtosis()
df_med['LDL L'].describe()

HDL = []
for genero, colesterol in zip(df_med['Género'], pd.to_numeric(df_med['Colesterol HDL'], errors='coerce')):
    if colesterol < 20 or colesterol > 210:
        HDL.append(None)
    elif genero == 'M' and colesterol < 40:
        HDL.append(5)
    elif genero == 'F' and colesterol < 50:
        HDL.append(5)
    elif genero == 'M' and colesterol >= 40 and colesterol < 60:
        HDL.append(1)
    elif genero == 'F' and colesterol >= 50 and colesterol < 60:
        HDL.append(1)
    elif colesterol >= 60:
        HDL.append(1)
    else:
        HDL.append(0)

df_med['HDL L'] = HDL
barplot1(df_med['HDL L'], "Distribución de Fecuencias HDL")
piechart1(df_med['HDL L'])
#Asimetria negativa casi simetrica
df_med['Control HTA'].skew()
#platicúrtica
df_med['Control HTA'].kurtosis()
df_med['Control HTA'].describe()

riesgo = []
for rcv in df_med['Clasificación del Riesgo Cardiovascular']:
    if rcv == "Leve":
        riesgo.append(1)
    elif rcv == "Moderado":
        riesgo.append(3)
    elif rcv == "Alto":
        riesgo.append(5)
    elif rcv == "Muy Alto":
        riesgo.append(7)
    else:
        riesgo.append(0)

df_med['Riesgo Cardiovascular L'] = riesgo
barplot1(df_med['Riesgo Cardiovascular L'], "Riesgo Cardiovascular L")
piechart1(df_med['Riesgo Cardiovascular L'])
#Asimetria positiva
df_med['Riesgo Cardiovascular L'].skew()
#leptocúrtica
df_med['Riesgo Cardiovascular L'].kurtosis()
df_med['Riesgo Cardiovascular L'].describe()


df_med['Puntaje Riesgo'] = df_med['Riesgo Cardiovascular L'] + df_med['HDL L'] + df_med['LDL L'] + df_med['Control HTA'] +  df_med['HbA1c'] + df_med['HTA y DM']

sns.distplot(df_med['Puntaje Riesgo'])
#Asimetria positiva
df_med['Puntaje Riesgo'].skew()
#platicúrtica
df_med['Puntaje Riesgo'].kurtosis()
df_med['Puntaje Riesgo'].describe()

df_med.sort_values(by=['Puntaje Riesgo'], inplace=True, ascending=False)
df_med['Puntaje Riesgo'].value_counts().plot(kind='bar', stacked=True)
df_med['Puntaje Riesgo'].value_counts()
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter(os.path.join(home, 'puntaje_riesgo.xlsx'), engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
df_med.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
writer.save()

# outlyers

ranges0 = [0,6,18,24,30,36]
plt.xticks([1,2,3,4,5]) 
df_med['Puntaje Riesgo'].groupby(pd.cut(df_med['Puntaje Riesgo'], ranges0)).count().plot(kind='bar')
z = df_med['Puntaje Riesgo'].groupby(pd.cut(df_med['Puntaje Riesgo'], ranges0)).count()
sns.set(style="darkgrid")
sns.barplot(z, alpha=0.9)
plt.xticks([1,2,3,4,5])    
plt.title("Rangos de Riesgo")
plt.ylabel('Número de ocurrencias', fontsize=12)
plt.xlabel('Riesgo', fontsize=12)
plt.show()

a = pd.to_numeric(df_med["Hemoglobina glicosilada"], errors='coerce')
rangesa = [0,4,20,a.max()]
a.groupby(pd.cut(a, rangesa)).count()
x = df_med["Hemoglobina glicosilada"].value_counts()
sum(n < 0 for n in a.values.flatten())


b = pd.to_numeric(df_med['Presión arterial sistólica'], errors='coerce')
rangesb = [0,90,250,b.max()]
b.groupby(pd.cut(b, rangesb)).count()
x1 = df_med['Presión arterial sistólica'].value_counts()
sum(n < 0 for n in b.values.flatten())



c = pd.to_numeric(df_med['Presión arterial diastólica'], errors='coerce')
rangesc = [0,60,140,c.max()]
c.groupby(pd.cut(c, rangesc)).count()
x2 = df_med['Presión arterial diastólica'].value_counts()
sum(n < 0 for n in c.values.flatten())

e = pd.to_numeric(df_med['Colesterol HDL'], errors='coerce')
rangese = [0,20,110,e.max()]
e.groupby(pd.cut(e, rangese)).count()
x4 = df_med['Colesterol HDL'].value_counts()
sum(n < 0 for n in e.values.flatten())

g = pd.to_numeric(df_med['Colesterol LDL'], errors='coerce')
rangesg = [0,70,250,g.max()]
g.groupby(pd.cut(g, rangesg)).count()
x6 = df_med['Colesterol LDL'].value_counts()
sum(n < 0 for n in g.values.flatten())


x8 = df_med['Estadio ERC'].value_counts()

good = df_med[(a.between(4, 20, inclusive=True)) & (b.between(90, 250, inclusive=True)) & (c.between(60, 140, inclusive=True)) & (e.between(20, 110, inclusive=True)) & (g.between(70, 250, inclusive=True))]

good.shape[0]

bad = df_med[(a < 4) & (a > 20) & (b < 90) & (b > 250) & (c < 60) & (c > 140) & (e < 20) & (e > 110) & (g < 70) & (g > 250)]

bad.shape


df1 = df_med[['HTA', 'DM', 'HbA1c', 'Control HTA', 'LDL L', 'HDL L', 'Riesgo Cardiovascular L', 'Puntaje Riesgo']]

corr = df1.corr()
print(corr)
sns.heatmap(corr, 
         xticklabels=corr.columns, 
         yticklabels=corr.columns)

df1["HTA"].fillna(0, inplace = True)
df1["DM"].fillna(0, inplace = True)
df1["HbA1c"].fillna(0, inplace = True)
df1["Control HTA"].fillna(0, inplace = True)
df1["LDL L"].fillna(0, inplace = True)
df1["HDL L"].fillna(0, inplace = True)
df1["Riesgo Cardiovascular L"].fillna(0, inplace = True)
df1["Puntaje Riesgo"].fillna(0, inplace = True)


df2 = pd.DataFrame()
df2["HTA"] = pd.Series(np.cbrt(df1['HTA'])).values
df2["DM"] = pd.Series(np.cbrt(df1['DM'])).values
df2["HbA1c"] = pd.Series(np.cbrt(df1['HbA1c'])).values
df2['Control HTA'] = pd.Series(np.cbrt(df1['Control HTA'])).values
df2['LDL L'] = pd.Series(np.cbrt(df1['LDL L'])).values
df2['HDL L'] = pd.Series(np.cbrt(df1['HDL L'])).values
df2['Riesgo Cardiovascular L'] = pd.Series(np.cbrt(df1['Riesgo Cardiovascular L'])).values
df2['Puntaje Riesgo'] = pd.Series(np.cbrt(df1['Puntaje Riesgo'])).values
df2.tail()
scaler = StandardScaler()
scaler.fit(df2)
df_normalized = scaler.transform(df2)
print(df_normalized.mean(axis = 0).round(2))
print(df_normalized.std(axis = 0).round(2))


barplot1(df1['HTA'], "Frecuencia de Diagnósticos HTA")
barplot1(df1['DM'], "Frecuencia de Diagnósticos DM")

dfTrain = df1[:22017]
dfTest = df1[22017:25384]
dfCheck = df1[25384:]

trainLabel = np.asarray(dfTrain['DM'])
trainData = np.asarray(dfTrain.drop('DM',1))
testLabel = np.asarray(dfTest['DM'])
testData = np.asarray(dfTest.drop('DM',1))

means = np.mean(trainData, axis=0)
stds = np.std(trainData, axis=0)
trainData = (trainData - means)/stds
testData = (testData - means)/stds

diabetesCheck = LogisticRegression()
diabetesCheck.fit(trainData, trainLabel)

accuracy = diabetesCheck.score(testData, testLabel)
print("accuracy = ", accuracy * 100, "%")

coeff = list(diabetesCheck.coef_[0])
labels = list(df1.drop('DM', 1).columns.values)
features = pd.DataFrame()
features['Features'] = labels
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Features', inplace=True)
features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')

joblib.dump([diabetesCheck, means, stds], 'diabeteseModel.pkl')
diabetesLoadedModel, means, stds = joblib.load('diabeteseModel.pkl')
accuracyModel = diabetesLoadedModel.score(testData, testLabel)
print("accuracy = ",accuracyModel * 100,"%")
print(dfCheck.head())

sampleData = dfCheck[:1]
# prepare sample
sampleDataFeatures = np.asarray(sampleData.drop('DM',1))
sampleDataFeatures = (sampleDataFeatures - means)/stds
# predict
predictionProbability = diabetesLoadedModel.predict_proba(sampleDataFeatures)
prediction = diabetesLoadedModel.predict(sampleDataFeatures)
print('Probability:', predictionProbability)
print('prediction:', prediction)


trainLabel = np.asarray(dfTrain['HTA'])
trainData = np.asarray(dfTrain.drop('HTA',1))
testLabel = np.asarray(dfTest['HTA'])
testData = np.asarray(dfTest.drop('HTA',1))

means = np.mean(trainData, axis=0)
stds = np.std(trainData, axis=0)
trainData = (trainData - means)/stds
testData = (testData - means)/stds

diabetesCheck = LogisticRegression()
diabetesCheck.fit(trainData, trainLabel)

accuracy = diabetesCheck.score(testData, testLabel)
print("accuracy = ", accuracy * 100, "%")

coeff = list(diabetesCheck.coef_[0])
labels = list(df1.drop('HTA', 1).columns.values)
features = pd.DataFrame()
features['Features'] = labels
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Features', inplace=True)
features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')

joblib.dump([diabetesCheck, means, stds], 'diabeteseModel.pkl')
diabetesLoadedModel, means, stds = joblib.load('diabeteseModel.pkl')
accuracyModel = diabetesLoadedModel.score(testData, testLabel)
print("accuracy = ",accuracyModel * 100,"%")
print(dfCheck.head())

sampleData = dfCheck[:1]
# prepare sample
sampleDataFeatures = np.asarray(sampleData.drop('HTA',1))
sampleDataFeatures = (sampleDataFeatures - means)/stds
# predict
predictionProbability = diabetesLoadedModel.predict_proba(sampleDataFeatures)
prediction = diabetesLoadedModel.predict(sampleDataFeatures)
print('Probability:', predictionProbability)
print('prediction:', prediction)
