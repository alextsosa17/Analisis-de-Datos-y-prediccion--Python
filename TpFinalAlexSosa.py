#LIBRERIAS A UTILIZAR 
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt 
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split #libreria para poder separar los datos entre entrenamiento y test
from sklearn.linear_model import LinearRegression #libreria para poder generar la regresion lineal 
from sklearn import metrics
#ANALISIS PREVIO 
"""
En el analisis previo con solo mirar el archivo, puedo deducir que es un archivo que contiene
edad de una persona, sexo de la persona solicitante del seguro,bmi(segun lo que googlie es un reembolso que se le puede dar), 
cuantos hijos tiene, si es fumador o no, y el precio del seguro a abonar.
De esto puedo inferir que mi prediccion va a ser con una regresion lineal multiple ya que la variable dependiete es el cargo del seguro, y es afectada
por todas las demas columnas, es decir, un seguro aumentara por todas las condiciones anteriores(edad, bmi, hijos, si es fumador o no y de que zona es la persona solicitante)
"""


#-------------GUARDO MI ARCHIVO A USAR -------------------------------
ruta =r'Unsam.Clase.12.4.6.2021\Unsam.Clase.12.4.6.2021\cvs_para_el_TP\seguro.csv'
dataSet = pd.read_csv(ruta) #archivo a utilizar 

#-----------------ANALISIS DE MIS DATOS ---------------------
print(dataSet)
#print("CANTIDAD DE FILAS Y COLUMNAS")
#print(dataSet.shape)
#no hace falta hacer el shape ya que al imprimir el dataSet, me dice cuantas filas y columnas tengo
print(dataSet.describe())

#compruebo si hay valores nan en mi codigo
print("Valores null o nan en el dataSet: ")
print(dataSet.isna().values.any())
print(dataSet.isnull().values.any())
#devuelve falso por ende no tengo valores NaN o null


#-------------SEPARO LAS VARIABLES QUE VOY A USAR------------
X = dataSet[['age','sex','bmi','children','smoker','region' ]].values
Y = dataSet['charges'].values 

print("X:\n")
print(X) #matriz     /variables independientes
print("Y:\n")
print(Y) #vector    /variables dependientes


#----------------COMIENZO A PREPARAR LOS DATOS-------------
"""
La edad y el bmi, son valores numericos ya manipulables , por ende , los dejo como estan para poder realizar mi analisis
"""

#MODIFICO LA COLUMNA DEL SEXO PARA MANEJARME CON 1 Y 0 
labelencoder_X=LabelEncoder()
X[:,1]=labelencoder_X.fit_transform(X[:,1])

#MODIFICO LA COMLUNMA DE SI ES FUMADOR CON 1 Y 0 
labelencoder_X=LabelEncoder()
X[:,4]=labelencoder_X.fit_transform(X[:,4])

#MODIFICO LA COLUMNNA DE SUR Y NORTE (CON ORIENTACION ESTE Y OESTE)
labelencoder_X=LabelEncoder()
X[:,-1]=labelencoder_X.fit_transform(X[:,-1])
print("Matriz X luego de preparacion de datos : \n")
print(X)

#PREGUNTAR SI HAY QUE ESCALAR LA EDAD?


#-------------GRAFICO MI VARIABLE DEPENDIENTE PARA SACAR ALGUNA CONCLUSION -----------

plt.figure(figsize=(10,5))
plt.tight_layout()
plt.title('Densidad de mi variable Dependiente')
seabornInstance.distplot(dataSet['charges'], color = 'lightblue')
plt.show()
"""
De este Grafico podemos concluir que mi variable dependiente 'charges' que seria el precio a pagar del seguro,
varia entre 0 y 70000 , y la mayor densidad(mayor cantidad de match) las hace desde el 0 hasta el 10000 aproximadamente.
"""

#------------------DIVIDIR LOS DATOS EN ENTRENAMIENTO Y TEST -----------------
#80 porciento de mis datos en entrenamiento y 20 en test
X_train , X_test, Y_train, Y_test = train_test_split(X , Y , test_size = 0.2, random_state=0)
#comienzo a entrenar mi modelo
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


#------------------ELIMINO MI COLUMNA QUE ES LA DE CARGOS(VARIABLE DEPENDIENTE)-----------
dataFrame= dataSet.drop(['charges'], axis = 1) #dropeo(elimino) mi variale dependiente de mi data frame
dataFrame = dataFrame.T #transpongo filas por columnas PODRIA SER TRANSPOSE()
dataFrame = dataFrame.index #me guardo en mi dataframe solo las etiquetas 
print(dataFrame) #imprimo las etiquetas para verificar que estan bien guardadas


#-----------------ENCONTRAR LOS COEFICIENTES MAS OPTIMOS PARA MIS ATRIBUTOS----------------

coeficiente_dataFrame = pd.DataFrame(regressor.coef_, dataFrame, columns= ['Coeficiente'])
print(coeficiente_dataFrame)
 #luego de imprimir concluyo que transpuso filas por columnas y tiene sus coeficentes correspondientes

#---------------PREDICCION--------------------

y_prediccion = regressor.predict(X_test)
#agarro mi dataFrame donde estan mis indices 
dataFrame = pd.DataFrame({'Actual': Y_test, 'Prediccion': y_prediccion})
dataFrame_prediccion = dataFrame.head(30)
print(dataFrame_prediccion)


#-------------------GRAFICO MI PREDICCION -----------------

dataFrame_prediccion.plot(kind='bar',figsize=(10,8)) #LE DIGO QUE ES UN GRAFICO DE BARRAS Y EL TAMAÑO 
plt.title('Actual vs Prediccion')
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


#----------------ME FIJO EL RENDIMIENTO DE MI ALGORITMO ---------------

print('Promedio de Error Absoluto:', metrics.mean_absolute_error(Y_test, y_prediccion)) 
print('Promedio de Error de Raiz:', metrics.mean_squared_error(Y_test, y_prediccion)) 
print('Error cuadratico medio de la raiz:', np.sqrt(metrics.mean_squared_error(Y_test, y_prediccion)))

#CONCLUSION FINAL
"""
Por el analisis hecho, podemos deducir que la prediccion no es nada buena, ya que los promedios de error
superan ampliamente el 10 porciento, es aproximadamente, un 15 porciento (siendo generoso), no es un analisis muy exacto, pero para nuestro caso
es aceptable(porque use regresion lineal multiple, entonces entre mas datos afecten a mi variable dependiente , menos me voy a acercar a una prediccion acertada),
ya que las columnas en las que supera ese porcentaje son la 14 , la 20 y la 23, que siendo generoso tendrian aproximadamente un 25 porciento extra.
"""


"""
PD: Perdon por mi manera de hablar o escribir pero la verdad que en las conclusiones tarde mas que en el codigo jajaja.
    Si hay algo que deba cambiar o mejorar porfavor diganmelo
Muchas gracias por TODO
Atte : Alex Sosa :)
"""