# Proyecto de Aprendizaje Automático

### _Realizado por Samuel Cardenete Rodríguez y Juan José Sierra González_

En este proyecto se estudia cómo hacer un adecuado preprocesamiento de datos para construir modelos lineales que resuelvan de la forma más óptima posible el problema.

Para esta práctica se utilizan dos bases de datos, una de clasificación y otra de regresión:
- **South African Heart Disease** es una base de datos de clasificación que trata de averiguar si una persona sudafricana tendrá o no una enfermedad cardiaca en función de algunos parámetros correspondientes a su estilo de vida.
- **Los Angeles Ozone** es una base de datos de regresión donde buscamos predecir la cota máxima por hora de la concentración de ozono de la ciudad de Upland, California.

Los datos son preprocesados utilizando el paquete _regsubsets_ y _PCA_, gracias al cual somos capaces de decidir qué características es mejor juntar para obtener un buen modelo. Probando distintas combinaciones de las características más recomendables y realizando alguna transformación sobre ellas pueden obtenerse mejores resultados. Se comparan los porcentajes de error entre ellos y se intenta encontrar el mejor modelo posible para cada uno de los dos problemas.
