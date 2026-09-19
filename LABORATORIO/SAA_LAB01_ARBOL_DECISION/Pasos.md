Claro. Te lo paso directamente como **código Markdown**, para que puedas copiarlo y guardarlo como:

`SAA_LAB01_ARBOLDECISION.md`

````markdown
# LAB01 - Árbol de Decisión: Predicción de abandono de clientes

**Empresa:** NexoTech Solutions  
**Archivo:** `SAA_LAB01_ARBOLDECISION.ipynb`

## Objetivo

NexoTech Solutions quiere identificar qué clientes tienen mayor probabilidad de abandonar la compañía (*churn*).

En esta práctica utilizaréis un **Árbol de Decisión** para:

- preparar un conjunto de datos;
- explorar los datos;
- separar variables predictoras y variable objetivo;
- dividir los datos en entrenamiento y prueba;
- entrenar un modelo;
- realizar predicciones;
- evaluar su funcionamiento;
- visualizar el árbol;
- interpretar las decisiones desde el punto de vista empresarial.

---

# PASO 1. Crear el Notebook

Dentro de la carpeta del laboratorio:

```text
SAA_LAB01_ARBOL_DECISION
```

cread/abrid:

```text
SAA_LAB01_ARBOLDECISION.ipynb
```

El Notebook debe quedar organizado mediante **celdas Markdown y celdas de código**.

---

# PASO 2. Título de la práctica

### Celda Markdown

```markdown
# SAA_LAB01 - Árbol de Decisión: Predicción de abandono de clientes

**Empresa:** NexoTech Solutions

**Objetivo:** utilizar un árbol de decisión para predecir qué clientes pueden abandonar la compañía (`churn`).
```

---

# PASO 3. Contexto del problema

### Celda Markdown

```markdown
## 1. Contexto

NexoTech Solutions es una empresa que ofrece servicios de telecomunicaciones.

La empresa quiere anticiparse al abandono de sus clientes.

Para ello dispone de información sobre:

- la permanencia del cliente;
- el número de llamadas realizadas al soporte técnico;
- la cuota mensual.

El objetivo es construir un modelo de Machine Learning capaz de predecir si un cliente abandonará la compañía.
Para ello utilizaremos un **Árbol de Decisión**.

El flujo de trabajo será:

**Datos → Exploración → Preparación → Entrenamiento → Predicción → Evaluación → Interpretación**
```

---

# PASO 4. Importar las librerías

### Celda de código

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

sns.set_theme(style="whitegrid")
```

### ¿Qué estamos haciendo?

- `numpy`: trabajar con datos numéricos y generar datos.
- `pandas`: crear y analizar el DataFrame.
- `matplotlib`: representar gráficos.
- `seaborn`: mejorar las visualizaciones.
- `scikit-learn`: construir y evaluar el modelo.

---

# PASO 5. Generar los datos

En esta práctica **no vamos a proporcionar un CSV**.

Los datos se generarán directamente en el Notebook para que todos los alumnos puedan reproducir la práctica.

### Celda Markdown

```markdown
## 2. Creación del conjunto de datos

Vamos a generar un conjunto de datos sintético que representa a clientes de NexoTech Solutions.

Las variables serán:

- `permanencia_meses`: meses que lleva el cliente en la compañía.
- `soporte_tecnico_calls`: número de llamadas al soporte técnico.
- `cuota_mensual`: cuota mensual del cliente.
- `churn`: indica si el cliente ha abandonado la compañía.

En `churn`:

- `0` → el cliente no abandona.
- `1` → el cliente abandona.
```

### Celda de código

```python
np.random.seed(42)

n_samples = 200

data = pd.DataFrame({
    "permanencia_meses": np.random.randint(1, 48, n_samples),
    "soporte_tecnico_calls": np.random.randint(0, 10, n_samples),
    "cuota_mensual": np.random.uniform(30, 120, n_samples)
})

data["churn"] = (
    (data["soporte_tecnico_calls"] > 4) |
    (data["permanencia_meses"] < 6)
).astype(int)

data.head()
```

### Explicación

Aquí estamos creando **200 clientes ficticios**.

La variable:

```python
churn
```

es la que queremos predecir.

Por tanto:

```text
Variables de entrada → permanencia, llamadas, cuota
                     ↓
              Árbol de decisión
                     ↓
              Predicción churn
```

---

# PASO 6. Explorar los datos

### Celda Markdown

```markdown
## 3. Exploración de los datos

Antes de entrenar un modelo debemos conocer los datos con los que vamos a trabajar.
```

### 6.1 Primeros registros

```python
data.head()
```

Deben observar las primeras filas.

### 6.2 Número de filas y columnas

```python
data.shape
```

Debería aparecer:

```text
(200, 4)
```

Tenemos:

- 200 clientes.
- 4 variables.

### 6.3 Nombres de las columnas

```python
data.columns
```

Deberán identificar:

```text
permanencia_meses
soporte_tecnico_calls
cuota_mensual
churn
```

### 6.4 Información del DataFrame

```python
data.info()
```

Aquí deberán comprobar:

- número de registros;
- nombres de las columnas;
- tipos de datos;
- ausencia de valores nulos.

### 6.5 Estadísticas descriptivas

```python
data.describe()
```

### Preguntas que deben responder

En una celda Markdown:

```markdown
### Análisis inicial

1. ¿Cuántos clientes tiene el conjunto de datos?
2. ¿Cuál es la permanencia mínima y máxima?
3. ¿Cuántas llamadas al soporte técnico puede tener un cliente?
4. ¿Cuál es aproximadamente la cuota mensual media?
5. ¿La variable `churn` utiliza valores 0 y 1?
```

---

# PASO 7. Separar X e y

### Celda Markdown

```markdown
## 4. Variables predictoras y variable objetivo

Para entrenar el modelo debemos separar:

- **X** → variables que utilizaremos para realizar las predicciones.
- **y** → variable que queremos predecir.
```

### Celda de código

```python
X = data[
    [
        "permanencia_meses",
        "soporte_tecnico_calls",
        "cuota_mensual"
    ]
]

y = data["churn"]
```

Después:

```python
X.head()
```

y:

```python
y.head()
```

### Pregunta

```markdown
**¿Cuál es la variable objetivo del modelo?**
```

Respuesta esperada:

> `churn`, porque queremos predecir si un cliente abandona o no.

---

# PASO 8. Dividir los datos

### Celda Markdown

```markdown
## 5. División de los datos

No debemos entrenar y evaluar el modelo utilizando exactamente los mismos datos.

Por ello dividiremos el conjunto en:

- 80 % → entrenamiento.
- 20 % → prueba.

Utilizaremos `random_state=42` para poder reproducir los resultados.
```

### Celda de código

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

Ahora comprobad los tamaños:

```python
print("Datos de entrenamiento:", X_train.shape)
print("Datos de prueba:", X_test.shape)
```

Deberían obtener aproximadamente:

```text
Datos de entrenamiento: (160, 3)
Datos de prueba: (40, 3)
```

---

# PASO 9. Crear el Árbol de Decisión

### Celda Markdown

```markdown
## 6. Creación del modelo

Utilizaremos un algoritmo de clasificación denominado Árbol de Decisión.

Limitaremos la profundidad del árbol a 3 niveles para evitar que el modelo sea excesivamente complejo.
```

### Celda de código

```python
clf_tree = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)
```

### Pregunta

¿Por qué utilizamos:

```python
max_depth=3
```

?

Debéis explicar que limitar la profundidad ayuda a evitar un árbol demasiado complejo y reduce el riesgo de **sobreajuste (overfitting)**.

---

# PASO 10. Entrenar el modelo

### Celda Markdown

```markdown
## 7. Entrenamiento

Ahora entrenaremos el árbol utilizando los datos de entrenamiento.
```

### Celda de código

```python
clf_tree.fit(X_train, y_train)
```

Aquí es donde el algoritmo **aprende las relaciones existentes en los datos**.

---

# PASO 11. Realizar predicciones

### Celda Markdown

```markdown
## 8. Predicciones

Una vez entrenado el modelo, utilizaremos los datos de prueba para comprobar qué predice.
```

### Celda de código

```python
y_pred = clf_tree.predict(X_test)
```

Podemos visualizar algunas predicciones:

```python
print(y_pred[:10])
```

Y compararlas con los valores reales:

```python
print(y_test.values[:10])
```

---

# PASO 12. Calcular la Accuracy

### Celda Markdown

```markdown
## 9. Evaluación del modelo

La primera métrica que utilizaremos será la Accuracy.

La Accuracy indica qué porcentaje de las predicciones realizadas por el modelo son correctas.
```

### Celda de código

```python
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy en Test Set: {acc * 100:.2f}%")
```

### Pregunta

```markdown
### Interpretación

¿Qué significa la Accuracy obtenida?

Explica el resultado utilizando tus propias palabras.
```

No basta con escribir:

> "La accuracy es del X %."

Debéis explicar qué significa ese porcentaje en el contexto de **NexoTech**.

---

# PASO 13. Matriz de confusión

### Celda Markdown

```markdown
## 10. Matriz de confusión

La matriz de confusión permite analizar con mayor detalle los aciertos y errores del modelo.
```

### Celda de código

```python
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.title("Matriz de Confusión - Árbol de Decisión")
plt.xlabel("Predicción")
plt.ylabel("Realidad")

plt.show()
```

---

# PASO 14. Interpretar la matriz

Debéis identificar:

| | Predice No Churn | Predice Churn |
|---|---:|---:|
| **Real: No Churn** | TN | FP |
| **Real: Churn** | FN | TP |

### Preguntas

```markdown
### Interpretación de la matriz de confusión

1. ¿Cuántos clientes que realmente no abandonaron fueron clasificados correctamente?
2. ¿Cuántos clientes que realmente abandonaron fueron detectados correctamente?
3. ¿Cuántos abandonos no consiguió detectar el modelo?
4. ¿Qué tipo de error podría ser especialmente importante para NexoTech?
5. ¿Por qué podría ser importante detectar a un cliente antes de que abandone?
```

En un problema de *churn*, un **falso negativo** puede ser costoso: el modelo dice que el cliente no abandonará, pero finalmente abandona.

---

# PASO 15. Visualizar el árbol

### Celda Markdown

```markdown
## 11. Visualización del Árbol de Decisión

Una de las ventajas de los árboles de decisión es que podemos visualizar las reglas utilizadas para realizar las predicciones.
```

### Celda de código

```python
plt.figure(figsize=(16, 8))

plot_tree(
    clf_tree,
    feature_names=X.columns,
    class_names=["No Churn", "Churn"],
    filled=True,
    rounded=True
)

plt.title("Árbol de Decisión - NexoTech Solutions")

plt.show()
```

---

# PASO 16. Interpretar las reglas

Esta es una de las partes más importantes desde el punto de vista de Machine Learning.

### Celda Markdown

```markdown
## 12. Interpretación del modelo

Analiza el árbol obtenido y responde:

1. ¿Cuál es la primera variable que utiliza el árbol para tomar una decisión?

2. ¿Qué condición aparece en la raíz del árbol?

3. ¿Qué ocurre cuando un cliente tiene muchas llamadas al soporte técnico?

4. ¿Qué ocurre con los clientes que llevan pocos meses en la compañía?

5. ¿Qué variable parece tener mayor importancia en las primeras decisiones del árbol?

6. Escribe dos reglas de decisión que puedas obtener del árbol.
```

**Importante:** las reglas deben obtenerse leyendo el árbol que habéis generado. No debéis copiar reglas sin comprobarlas.

Una regla puede tener esta estructura:

> Si el número de llamadas al soporte técnico supera determinado valor, el árbol considera que aumenta la probabilidad de abandono.

Otra:

> Si la permanencia del cliente es inferior a determinado número de meses, el árbol puede clasificarlo como cliente con riesgo de abandono.

---

# PASO 17. Interpretación empresarial

### Celda Markdown

```markdown
## 13. Interpretación empresarial

NexoTech Solutions no necesita únicamente conocer la Accuracy del modelo.

La empresa quiere utilizar las predicciones para tomar decisiones.

### A. Soporte técnico

¿Qué relación observas entre el número de llamadas al soporte técnico y el abandono?

### B. Permanencia

¿Qué relación observas entre la permanencia del cliente y el abandono?

### C. Acción empresarial

Imagina que NexoTech recibe una predicción indicando que un cliente tiene riesgo de abandono.

¿Qué acción podría realizar la empresa?

Propón al menos dos acciones.
```

Por ejemplo:

- contactar con el cliente;
- ofrecer asistencia personalizada;
- revisar incidencias;
- ofrecer una promoción;
- mejorar el servicio.

---

# PASO 18. Conclusiones

Al final del Notebook:

```markdown
# 14. Conclusiones

Después de realizar la práctica, responde:

1. ¿Qué problema de Machine Learning hemos resuelto?

2. ¿Es un problema de clasificación o regresión? ¿Por qué?

3. ¿Cuál es la variable objetivo?

4. ¿Qué variables utiliza el modelo para realizar las predicciones?

5. ¿Qué porcentaje de los casos ha clasificado correctamente?

6. ¿Qué información aporta la matriz de confusión que no nos proporciona únicamente la Accuracy?

7. ¿Qué reglas del árbol pueden ser útiles para NexoTech?

8. ¿Qué limitaciones tiene este modelo?

9. ¿Utilizarías este modelo directamente en una empresa real? Justifica tu respuesta.
```

---

# PASO 19. Entrega

El alumno deberá entregar:

```text
SAA_LAB01_ARBOLDECISION.ipynb
```

El Notebook debe contener:

- código ejecutado;
- resultados;
- gráficos;
- matriz de confusión;
- árbol de decisión;
- respuestas a las preguntas;
- interpretación empresarial;
- conclusiones.

**No se valorará únicamente que el código funcione.**

Se valorará especialmente que el alumno sea capaz de pasar de:

**Datos → Modelo → Predicción → Evaluación → Interpretación → Decisión empresarial**

---

# Resumen del flujo de trabajo

```text
1. Crear los datos
       ↓
2. Explorar los datos
       ↓
3. Separar X e y
       ↓
4. Dividir entrenamiento / prueba
       ↓
5. Crear el modelo
       ↓
6. Entrenar
       ↓
7. Predecir
       ↓
8. Evaluar
       ↓
9. Visualizar
       ↓
10. Interpretar
       ↓
11. Tomar conclusiones empresariales
```

## Objetivo final de la práctica

El objetivo no es simplemente aprender a utilizar `DecisionTreeClassifier`.

El objetivo es comprender el **flujo completo de un problema de Machine Learning**:

**Datos → Modelo → Predicción → Evaluación → Interpretación → Decisión empresarial**
````
