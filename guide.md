
---

# **Guía - Reinforcement Learning**
- Elaborado por: Miguel Pérez

---
# Tipos de Datos

### **Datos Discretos (enteros)**
- Valores contables  
- Ej: edad, número de fallas

### **Datos Categóricos**
- **Binomiales:** 2 categorías (sí/no, hombre/mujer)  
- **Multinomiales:** >2 categorías (tipo de combustible, color de ojos)

### **Datos Continuos**
- Valores en un intervalo infinito  
- Ej: temperatura, humedad, velocidad, voltaje

---

## Histogramas
- Representan **frecuencias** de valores o rangos.  
- Sirve para:
  - Entender distribución  
  - Ver patrones  
  - Detectar valores comunes o raros  

---

# Distribuciones

### **Distribución Uniforme**
- Todos los valores tienen la MISMA probabilidad.  
- Ej: dado justo, baraja mezclada.

### **Distribución Binomial**
- Solo dos resultados posibles: éxito / fracaso.  
- Ej: lanzar una moneda n veces.

### **Distribución Multinomial**
- Más de dos categorías excluyentes.  
- Ej: salidas de las caras de un dado.

### **Distribución Normal (Gaussiana)**
- Forma de campana.  
- Media = mediana = moda  
- Común en fenómenos naturales.

### **Distribución Poisson**
- Cuenta eventos en un intervalo.  
- Ej: llamadas por minuto, gotas por segundo.

### **Distribución Pareto**
- Describe fenómenos 80-20.  
- Ej: riqueza, producción, ventas.

### **Distribución Beta**
- Variable continua entre 0 y 1.  
- Modela proporciones.

---

# Medidas Estadísticas

### **Moda**
Valor que más se repite.

### **Media**
$$
\text{mean} = \frac{\sum x}{n}
$$

### **Mediana**
Valor central del conjunto ordenado.

### **Varianza**
Como estan dispersos los datos en una distribución.
$$
\sigma^2 = \frac{\sum (x-\mu)^2}{n}
$$

### **Desviación Estándar**
Indica los rangos en los que los datos se presentan en mayor proporción.
$$
\sigma = \sqrt{\text{Var}}
$$


---

# Teorema Central del Límite (CLT)

- Si tomas **muchas muestras** de cualquier distribución,
  la **distribución de sus medias será Normal**.
- Usado en:
  - Ciencia
  - Encuestas
  - ML: Policy Gradients, métodos con expected value

---

# Probabilidad

$$
p(x) = \frac{\text{veces que ocurre}}{\text{total de eventos}}
$$

---

# PDF — Probability Density Function

Each distribution type has a function called the Probability Density Function (PDF) which intends to model the density of a given dataset and return a number between 0 and 1 that signals how dense the data is. Each distribution has its own PDF equation.

## **PDF Gaussiana / Discretos**
$$
\mu = mean
$$
$$
\sigma = Std Dev
$$
$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

## **PDF Binomial / Categórica**

$$
\beta = Spread (modifies\;sigmoid-like \;curve)
$$
$$a_1, a_2, a_N = parameters$$

$$
P(a_1, a_2, \beta) =  \sigma(\beta(a_1-a_2)) = \frac{1}{1+e^{-x}}
$$

$$
P(a_1, a_2,...,a_N) =  \frac{e^{ a_i }}{\sum e^{ a_j }}
$$



---

# CDF — Cumulative Distribution Function

Da la probabilidad de que:  
$$
X \leq x
$$

## Outliers (IQR)
$$
\text{IQR} = Q3 - Q1
$$

Outlier si:
$$
x > Q3 + 1.5(IQR)
$$
o  
$$
x < Q1 - 1.5(IQR)
$$

---

# Valor Esperado (Expected Value)

## Sin pesos:
$$
E[X] = \frac{1}{N}\sum x_i
$$

## Con pesos/p(x):
Average of data transformed by a function (e.g.
log(x)) weighed by its likelihood (i.e. p(x))
$$
E[f(X)] = \sum p(x_i)\, f(x_i)
$$

---

# Multi-Armed Bandit (Tragamonedas)

Actualización incremental del valor esperado:

$$
Q_k = Q_{k-1} + \frac{1}{k}(r_k - Q_{k-1})
$$

---

# Exploration vs Exploitation

## **Epsilon-Greedy**
$$
\epsilon_t = \epsilon_{end} + (\epsilon_{start}-\epsilon_{end}) e^{-t/\text{decay}}
$$

- random > ε → explotación  
- random ≤ ε → exploración  

## **Softmax (Boltzmann) o Sigmoid para muestreo**
$$
P(a) = \frac{e^{\beta Q(a)}}{\sum e^{\beta Q(a')}}
$$
$$
P(Q, \beta) =  \sigma(\beta(Q_a-Q_b)) = \frac{1}{1+e^{-\beta (Q_a-Q_b)}}
$$
β parámetro de control (“temperature”) controla qué tan “determinista” es la elección.

---

# Reglas de Probabilidad

## **Suma ( two mutually exclusive events happening):**
$$
P(A \cup B) = P(A) + P(B)
$$

## **Producto ( both events happening ):**
$$
P(A \cap B) = P(A)P(B)
$$

## **Probabilidad Condicional**
Compute the probability of A
given the ocurrence of B. This means that
B must happen first, subject to its own
uncertainty, and only then, from what is
left, A can happen with a given
probability
$$
P(A|B) = \frac{P(A\cap B)}{P(B)}
$$

Base del Teorema de Bayes.

---

#  Maximum Likelihood — Apuntes

## Probabilidad vs Likelihood
- **Probabilidad**: dado un modelo (parámetros conocidos), ¿qué tan probable es que X tome ciertos valores?  
- **Likelihood (verosimilitud)**: dada una observación $x$ y una familia de distribuciones parametrizadas por $\theta$, la verosimilitud es:

$$
L(\theta) = f(x \mid \theta)
$$

## PDF normal (Gaussiana)
$$
f(x \mid \mu, \sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

## Maximum Likelihood
Para datos $x_1,\dots,x_n$ y parámetros $\theta$:

$$
L(\theta) = \prod_{i=1}^n f(x_i \mid \theta), \qquad
\ln L(\theta) = \sum_{i=1}^n \ln f(x_i \mid \theta)
$$

Use logarithms to simplify computations and make use of its concave property

## Distribución Normal donde $\theta=[\mu,\sigma^2]$

Log-likelihood:
$$
\ln L(\mu,\sigma^2) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2
$$

Derivando parcialmente e igualando a cero:

- Estimador batch de la media (MLE):
  $$
  \hat{\mu} = \frac{1}{n}\sum_{i=1}^n x_i
  $$

- Estimador batch de la varianza (MLE):
  $$
  \hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \hat{\mu})^2
  $$

## Optimización por gradiente
- Calcula $\nabla_\theta \ln L(\theta)$ y usa descenso por gradiente:

  $$
  \theta \leftarrow \theta + \eta \nabla_\theta \ln L(\theta)
  $$

  (o la variante con signo negativo si minimizas la *neg-log-likelihood*).

## Estimación secuencial (online)
- Media incremental:
  $$
  \hat{\mu}_{t+1} = \hat{\mu}_t + \frac{1}{N}(x_{t+1} - \hat{\mu}_t)
    $$
- Varianza incremental (forma simple mostrada):
  $$
  \hat{\sigma}^2_{t+1} = \hat{\sigma}^2_t + \frac{(x_{t+1} - \hat{\mu}_t)^2 - \hat{\sigma}^2_t}{t+1}
  $$
  $$
  \hat{\sigma}^2_{t+1} = \hat{\sigma}^2_t + \frac{1}{N}((x_t - \hat{\mu}) - \hat{\sigma}^2_t)
  $$
  
---

# Multinomial Distribution Approximation
## Multinomial PDF

Sea un conjunto de parámetros $a_k$, donde $a_j$ es el parámetro asociado a la categoría cuya probabilidad queremos modelar.

La probabilidad de la categoría $j$ está dada por:

$$
S_j = \frac{e^{\beta a_j}}{\sum_{i=1}^{n} e^{\beta a_i}}
$$

donde $\beta$ es un parámetro de control (similar al "inverse temperature").

---

## Optimization (Maximum Likelihood)

Queremos ajustar los parámetros $\{a_1, a_2, ..., a_k, \beta\}$ para aproximar correctamente la distribución multinomial.

La PDF general:

$$
\text{pdf}(x \mid \theta) = \frac{e^{\beta a_x}}{\sum_{i=1}^{n} e^{\beta a_i}}
$$

La likelihood del dataset:

$$
L(x_1,\dots,x_n \mid \theta) = \prod_{i=1}^{n} \text{pdf}(x_i\mid\theta)
$$

### Derivadas parciales (softmax)

Derivada del score $S_j$ respecto a su parámetro:

- **Cuando $i = j$**:

$$
\frac{\partial S_j}{\partial a_j} = S_j(1 - S_j)
$$

- **Cuando $i \neq j$**:

$$
\frac{\partial S_j}{\partial a_i} = -S_j S_i
$$

Estas son exactamente las derivadas del **softmax** estándar.

---

# Optimization via Policy Learning

En un agente RL, si cada acción corresponde a una categoría de la distribución multinomial, la política viene dada por:

$$
p(a_i) = \frac{e^{\beta a_i}}{\sum_{k=1}^n e^{\beta a_k}}
$$

El **valor esperado del retorno** es:

$$
\bar{r} = \sum_{i=1}^n p(a_i)\, r_i
$$

---

## Gradient of Expected Reward

El gradiente del valor esperado respecto al parámetro $a_i$ es:

$$
\frac{\partial \bar{r}}{\partial a_i}
= \beta\, p(a_i)\,\left(r_i - \bar{r}\right)
$$

Esta es la forma clásica:

> **Policy Gradient = Softmax Gradient × Advantage**

donde el *advantage* es $r_i - \bar{r}$.

---

## Parameter update rules

### Regla de actualización general:

$$
a_i \leftarrow a_i + \lambda\, \frac{\partial \bar{r}}{\partial a_i}
$$

Sustituyendo el gradiente:

$$
a_i \leftarrow a_i + \lambda \beta\, p(a_i)\,(r_i - \bar{r})
$$

---



## Alternativa por casos (como en la presentación)

Cuando la acción tomada es $i$:

$$
a_i \leftarrow a_i + \lambda (1 - p(a_i))(r_i - \bar{r})
$$

Para todas las acciones no tomadas ($j \neq i$):

$$
a_j \leftarrow a_j - \lambda\, p(a_j)(r_i - \bar{r})
$$

Estas fórmulas equivalen a la derivada del softmax *policy gradient*.

---

# Reinforcement Learning (RL)

---

## Introducción y Fundamentos

### Machine Learning: Comparativa
* **Supervised Learning:** Tenemos datos etiquetados por humanos (Input $\to$ Target).
* **Unsupervised Learning:** Tenemos datos, pero no etiquetas (buscamos patrones/estructuras).
* **Reinforcement Learning:** **No tenemos datos previos**. Tenemos un **agente** y un **entorno** que provee **recompensas**.

### Elementos del RL
El ciclo básico de interacción:
1.  **Agente:** Entidad artificial que analiza observaciones y emite acciones.
2.  **Entorno (Environment):** Sistema que recibe la acción, transiciona a un nuevo estado y emite una observación y una recompensa.
3.  **Recompensa (Reward):** Señal escalar que indica qué tan buena fue la acción con respecto a un objetivo.
4.  **Política (Policy):** La estrategia del agente (mapeo de observaciones a acciones).

> **El Problema Central:** Cómo observar, recolectar y analizar datos para emitir acciones que **maximicen la recompensa acumulada**.

### Tipos de Motivación
* **Extrínseca:** La recompensa es diseñada por humanos (ingeniería de recompensas) para guiar al agente (ej. puntos en un juego).
* **Intrínseca:** Señal generada por el propio agente para fomentar la exploración.
    * *Curiosidad:* Basada en el error de predicción (si no puedo predecir qué pasará, quiero ir ahí).
    * *Empowerment:* Capacidad de controlar el entorno.

---

## Tipos de Enfoques en RL

### Por Modelo
1.  **Model-Free (Libre de modelo):** Mapea observaciones directamente a acciones o valores usando prueba y error. No intenta entender "cómo funciona la física" del entorno.
2.  **Model-Based (Basado en modelo):**
    * Entrena un modelo para predecir la dinámica del entorno (Estado actual + Acción $\to$ Siguiente Estado).
    * Usa ese modelo para planificar o entrenar una política.

### Por Método de Aprendizaje
1.  **Value-based:** Aprende el valor numérico de estar en un estado o tomar una acción ($Q$). Elige la acción con mayor valor.
2.  **Policy-based:** Aprende directamente la función de probabilidad de las acciones dado un estado.
3.  **Actor-Critic:** Híbrido. Un *Actor* decide la acción y un *Crítico* estima el valor de esa acción para ajustar al actor.

---

## Entornos y Gymnasium

### Tipos de Entornos
* **K-armed Bandits:** Tragamonedas. Elegir opciones con diferentes probabilidades de recompensa (sin estados secuenciales).
* **Mazes (Laberintos):** Espacio navegable con obstáculos y metas.
* **Robots:** Sistemas mecánicos (caminar, agarrar). Control motor continuo.
* **Juegos:** StarCraft, Atari. Usados para testear algoritmos (benchmarks).

### Estructura Gymnasium (Python Wrapper)
Librería estándar para entornos de RL. Clase principal `CustomEnv`:
* `__init__()`: Constructor, define variables iniciales.
* `reset()`: Restaura el entorno al inicio y devuelve la primera observación.
* `step(action)`: Ejecuta una acción. Retorna:
    * `observation`: Nuevo estado.
    * `reward`: Recompensa obtenida.
    * `terminated/truncated`: Booleano (¿terminó el juego?).

---

## Value-Based Methods (Métodos Basados en Valor)

### Concepto Biológico
Inspirado en la dopamina. Las neuronas refuerzan sinapsis cuando la recompensa recibida es mayor a la esperada (error de predicción positivo).

### Expected Value (Valor Esperado)
Es el promedio ponderado de los resultados posibles.
$$E[f] = \sum p(x_i) f(x_i)$$
* Donde $p(x_i)$ es la probabilidad de que ocurra el evento y $f(x_i)$ el valor del evento.

### Q-Learning (Tabular)
El cerebro/agente modela el valor esperado de las opciones.

**Actualización de Valor (Simple):**
$$Q_k = Q_{k-1} + \alpha (r_k - Q_{k-1})$$
* $Q_k$: Valor acumulado.
* $\alpha$ (o $1/k$): Tasa de aprendizaje (Learning Rate).
* $r_k$: Recompensa actual.
* **Interpretación:** El nuevo valor es el viejo valor más una fracción del "error" (diferencia entre lo que recibí y lo que creía que iba a recibir).

**Valor Relativo (Bavard et al.):**
El cerebro normaliza los valores basándose en el contexto (min y max recompensas disponibles).
$$Q_k = Q_{k-1} + \alpha \left( \frac{r_{obj} - r_{min}}{r_{max} - r_{min}} - Q_{k-1} \right)$$

### Temporal Difference (TD) y Ecuación de Bellman
Para decisiones secuenciales (donde el futuro importa).
**Ecuación Clave:**
$$Q(s, a)_{new} = Q(s, a)_{old} + \alpha \underbrace{[r + \gamma \cdot \max Q(s', a') - Q(s, a)_{old}]}_{\text{TD Error}}$$

* $\gamma$ (Gamma): Factor de descuento. Qué tanto me importa el futuro vs el presente.
* $\max Q(s', a')$: La mejor suposición del valor del *siguiente* estado.

---

## Markov Decision Process

Un MDP es un **caso especial de las Cadenas de Markov**.
* **Cadena de Markov normal:** Las transiciones ocurren de forma estocástica "porque sí" (fenómenos naturales).
* **MDP:** Las transiciones son provocadas por una **fuente externa** (Agente o Usuario). El sistema no cambia de estado a menos que se ejecute una acción ($a$).

### Ciclo de Interacción
1.  **Estado Actual:** El sistema está en un estado $S$.
2.  **Acción Externa:** El agente selecciona una acción ($a_{ij}$) de una matriz de acciones posibles.
3.  **Transición y Recompensa:** El sistema cambia al siguiente estado y "devuelve" una recompensa ($r_{ij}$).

### La Propiedad de Markov (The Markovian Assumption)
Es la regla de oro para que las matemáticas funcionen. Establece que el futuro es independiente del pasado, dado el presente.
> *"La probabilidad de pasar al siguiente estado y obtener una recompensa depende **únicamente** del estado actual y la acción tomada, no de la historia previa."*

$$P(S_{t+1} | S_t, a_t, S_{t-1}, ...) = P(S_{t+1} | S_t, a_t)$$

---

## Exploration vs Exploitation

El dilema: ¿Pruebo algo nuevo (aprender) o elijo lo que sé que funciona (ganar recompensa)?

1.  **Epsilon-Greedy ($\epsilon$-greedy):**
    * Tirar un dado. Si sale bajo ($\epsilon$), elijo una acción aleatoria (Exploración).
    * Si sale alto, elijo la mejor acción conocida (Explotación).
    * *Decay:* $\epsilon$ empieza alto (mucho random) y baja con el tiempo.

2.  **Softmax / Sigmoide:**
    * Convierte los valores $Q$ en probabilidades. Si una acción es mucho mejor, tiene mucha más probabilidad de ser elegida, pero no el 100%.
    * $$P(a) = \frac{e^{Q(a)/\tau}}{\sum e^{Q(b)/\tau}}$$

---

## Policy Gradients (Gradientes de Política)

### Objetivo
Optimizar directamente los parámetros ($\theta$) de la política $\pi$ para maximizar la recompensa total esperada ($J$).

Parametrizamos la política como $\pi_\theta(a|s)$ y queremos **maximizar** el retorno esperado sobre trayectorias $\tau$:

$$
J(\theta)=\mathbb{E}_{\tau\sim p_\theta(\tau)}\big[ R(\tau) \big]
$$

donde $R(\tau)=\sum_{t=0}^{T} r_t$. Aquí $p_\theta(\tau)$ es la probabilidad de la trayectoria bajo la política y la dinámica del entorno.

## 2) Log-derivative trick

Queremos $\nabla_\theta J(\theta)$.

$$
\begin{aligned}
\nabla_\theta J(\theta)
&= \nabla_\theta \int p_\theta(\tau)\, R(\tau)\, d\tau \\
&= \int \nabla_\theta p_\theta(\tau)\, R(\tau)\, d\tau
\end{aligned}
$$

Usamos el truco de derivada logarítmica:

$$
\nabla_\theta p_\theta(\tau)=p_\theta(\tau)\nabla_\theta \log p_\theta(\tau)
$$

Sustituyendo:

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau\sim p_\theta}\big[ \nabla_\theta \log p_\theta(\tau)\, R(\tau) \big]
$$

---

## 3) Factorización por pasos de tiempo

La probabilidad de una trayectoria:

$$
p_\theta(\tau)=p(s_0)\prod_{t=0}^{T} \pi_\theta(a_t|s_t)\, p(s_{t+1}|s_t,a_t)
$$

Tomando logaritmo:

$$
\log p_\theta(\tau)=\sum_{t=0}^{T} \log \pi_\theta(a_t|s_t) + \text{const}
$$

Por tanto, la Ecuación del Gradiente:

$$
\nabla_\theta J(\pi_\theta)=
\mathbb{E}_{\tau\sim p_\theta}
\left[
\sum_{t=0}^{T}
\nabla_\theta \log \pi_\theta(a_t|s_t)\,\cdot R(\tau)
\right]
$$

* **Interpretación:** Ajustamos los parámetros $\theta$ para hacer más probables las acciones $(a_t)$ que resultaron en una alta recompensa acumulada $R(\tau)$.

## Problemas Comunes en Policy Gradient
### 1. **Alta Varianza en el Gradiente**

Los métodos de Policy Gradient estiman el gradiente esperado de la recompensa:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s)\, R \right]
$$

El problema es que la estimación:

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N 
\nabla_\theta \log \pi_\theta(a_i|s_i) R_i
$$

tiene **varianza muy alta**, especialmente cuando:

- $R$ depende de trayectorias largas
- el espacio de estados es grande
- las políticas cambian demasiado entre actualizaciones

Esto provoca **inestabilidad**, actualizaciones ruidosas y aprendizaje lento.

---

### 2. **Exploración Ineficiente**

La política se actualiza en la dirección de acciones que han dado buena recompensa:

$$
\nabla_\theta \log \pi_\theta(a|s) R
$$

Si la política inicial es mala y produce pocas acciones con recompensa:

- el gradiente es pequeño
- la política no explora lo suficiente
- se queda atrapada en óptimos locales


---

### 3. **Sensibilidad a Hiperparámetros**

Especialmente al *learning rate*:

- si es muy pequeño → aprendizaje extremadamente lento  
- si es muy grande → divergencia

---

### Baseline: Resolver Problema de Varianza
Los gradientes puros son muy ruidosos.
* **Baseline:** Restar un valor base para reducir varianza de obtención de reward y evitar sobrepremitar una acción ineficiente. 
* **Advantage Function ($A$):**
    $$A(s, a) = Q(s, a) - V(s)$$
    * ¿Qué tanto mejor es esta acción comparada con el promedio de estar en este estado?

  
Existen tres variantes clásicas:


#### 1. Baseline Global: **Promedio de Rewards Obtenidos**

Se utiliza un valor escalar $b$ que promedia todos los rewards obtenidos en episodios recientes:

$$
b = \frac{1}{N}\sum_{i=1}^N R_i
$$

Gradiente actualizado:

$$
\nabla_\theta J = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a|s)(R - b) \right]
$$


#### 2. Baseline por Acción: **Promedio de Reward por Acción (Q-value)**

Aquí se usa como baseline el valor esperado de tomar una acción en un estado:

$$
b(s,a) = Q(s,a)
$$

El gradiente se vuelve:

$$
\nabla_\theta J = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a|s)(R - Q(s,a)) \right]
$$

Este baseline corresponde directamente a la idea de **estimadores del Q-value**.

#### 3. Advantage Baseline: **$Q(s,a)$ Menos Promedio Global de Rewards**

Esta variante mezcla las dos anteriores:

$$
b = \mathbb{E}[R], \qquad A(s,a) = Q(s,a) - b
$$

Entonces el gradiente utiliza:

$$
\nabla_\theta J = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a|s)\, A(s,a) \right]
$$

#### **Tabla Comparativa**

| Variante | Fórmula | Ventajas | Desventajas |
|---------|---------|----------|-------------|
| **Promedio global de rewards** | $R - b$ | Fácil, reduce varianza | Ignora acción/estado |
| **Promedio por acción (Q-value)** | $R - Q(s,a)$ | Modela calidad real de acciones | Costoso; depende del critic |
| **Q-value − promedio global** | $Q(s,a) - b$ | Combina ambas ventajas | Requiere estimar $Q$ correctamente |

---

## Actor-Critic y Algoritmos de Optimización Avanzados

Cuando entrenas una política estocástica, las acciones se eligen al azar según sus probabilidades.
A veces, por pura suerte, una acción mala puede recibir muchas recompensas positivas en una trayectoria específica.

¿Consecuencia?
El algoritmo de policy gradient ajusta la política para favorecer más esa acción mala, porque observa que “dio buen reward”, aunque en realidad no sea buena.

Esto genera que la política nueva cambie demasiado respecto a la anterior, inclinándose hacia acciones que aparentemente funcionaron, pero que no son realmente las mejores.

### Actor-Critic

El método **Actor-Critic** combina dos ideas clave:

1. **Actor ($\pi_\theta$):** Red neuronal que decide qué acción tomar. (Política) 
2. **Critic ($V_\phi$):** Red neuronal que estima el valor del estado ($V(s)$) para calcular el *Advantage* y reducir la varianza del gradiente.

Juntos permiten entrenar políticas estocásticas más estables que el método REINFORCE tradicional.

---

### ¿Por qué Actor-Critic?

El problema del Policy Gradient puro (REINFORCE) es la **alta varianza del término $R(\tau)$**, lo que causa actualizaciones ruidosas y aprendizaje lento.

Para solucionar esto:

- Se introduce un **Critic** que estima el valor esperado del estado $V(s)$.
- Este valor sirve como baseline para calcular el *Advantage*:
  
$$
A(s,a) = R(\tau) - V_\phi(s)
$$

Esto estabiliza el gradiente y acelera la convergencia.

---

### Entrenamiento del Critic (Estimador de $V_\phi$)

El Critic se entrena para aproximar la función de valor mediante regresión:

$$
V_\phi(s) \approx \mathbb{E}[R(\tau) \mid s]
$$

La actualización del Critic es:

$$
\phi = \phi + \nabla_\phi \left( \| \hat{V}_\phi(s) - R(\tau) \|_2^2 \right)
$$

Es decir:

- Se minimiza el **error cuadrático** entre la predicción del Critic y la recompensa real.
- Esto convierte al Critic en un baseline adaptativo que aprende con la experiencia.

---

### Entrenamiento del Actor (Actualización de Política)

El Actor ajusta los parámetros $\theta$ siguiendo un gradiente de política ponderado por el *Advantage* estimado:

$$
\theta = \theta + \nabla_\theta \log \pi_\theta(a \mid s)\, \left( R(\tau) - V_\phi(s) \right)
$$

Interpretación:

- Si una acción $a$ produjo un reward **mayor** que lo que esperaba el Critic → la política debe aumentar su probabilidad.
- Si produjo un reward **peor** de lo esperado → la política debe disminuir su probabilidad.

---

### ¿Qué está pasando matemáticamente?

**Critic:** intenta responder  
> “¿Qué tan bueno es este estado en general?”

**Actor:** intenta responder  
> “¿Debería repetir esta acción en estados similares?”

El *Advantage* los conecta:

$$
A(s,a) = R(\tau) - V_\phi(s)
$$

Con esto:

- **El Critic reduce la varianza** del gradiente.
- **El Actor recibe un gradiente más preciso y con menos ruido.**

Actor–Critic NO controla cuánto cambia la política entre updates.

El problema descrito originalmente es:

- La política nueva puede ser demasiado diferente a la vieja.

- Actor-Critic no tiene un mecanismo para limitar la magnitud del cambio de política.

---
### Trust Region & PPO

#### Problema principal
En *Policy Gradient* tradicional, el update sobre los parámetros $\theta$ puede ser tan grande que:
- la nueva política se aleja demasiado de la anterior,
- el agente “olvida” comportamientos útiles,
- el entrenamiento se vuelve inestable y puede **divergir**.

Para evitarlo nacen los **Métodos de Optimización por Regiones de Confianza** (Trust Region Methods):

### **TRPO — Trust Region Policy Optimization**

TRPO define una región matemática en el espacio de soluciones dentro de la cual los parámetros pueden moverse sin destruir la política previa.

#### 🔹 Idea central  
Limitar cuánto puede cambiar la política nueva respecto a la política vieja:
- usando una restricción de **KL Divergence**,
- para evitar saltos demasiado grandes.

La actualización maximiza un objetivo nuevo, **pero bajo la restricción**:

$$
D_{KL}(\pi_{\theta_{\text{old}}} \;\|\; \pi_{\theta_{\text{new}}}) \le \delta
$$

donde:
- $D_{KL}$ mide cuánta información cambia entre políticas,
- $\delta$ es un límite máximo permitido.

#### 🔹 Nuevo objetivo optimizado
TRPO maximiza una versión corregida del *surrogate objective*:

$$
L^{TRPO}(\theta) = 
\mathbb{E}\left[
\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A^{\pi_{\text{old}}}(s,a)
\right]
$$

pero únicamente permite updates $\theta$ tal que la divergencia sea pequeña.


#### Ventajas de TRPO
- Evita que la política cambie demasiado rápido.  
- Mantiene un comportamiento más estable que Policy Gradient vanilla.  
- Reduce el riesgo de colapsar la política hacia acciones malas.  

#### Desventaja
- **Computacionalmente costoso:**  
  Requiere resolver un problema de optimización con restricciones (método conjugado, hessianos aproximados, etc.).

Esto llevó a desarrollar un método más simple…

### **PPO — Proximal Policy Optimization**

PPO es una versión práctica de TRPO: mantiene la idea de limitar cuánto puede cambiar la política, pero evita la optimización costosa basada en KL Divergence. En su lugar, usa un mecanismo simple de **clipping** para restringir el tamaño del update.

---


#### Idea clave del método

Se calcula el **ratio** entre la política nueva y la antigua:

$$
r(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}
$$

PPO fuerza este ratio a permanecer cerca de 1.  
El límite se define con un hiperparámetro $\epsilon$ (típicamente $0.2$) y el objetivo que se maximiza es:

$$
L^{PPO}(\theta) =
\mathbb{E}\left[
\min\left(
r(\theta)A,\;
\text{clip}(r(\theta), 1-\epsilon, 1+\epsilon)A
\right)
\right]
$$

donde:

- $r(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$
- $A$ es el Advantage estimado para esa muestra.
##### 🔹 1. Cuando el ratio está dentro del rango permitido

Si:

$$
1 - \epsilon \le r(\theta) \le 1 + \epsilon
$$

entonces el gradiente es el mismo que en Policy Gradient:

$$
\nabla_\theta L =
A \, r(\theta)\, \nabla_\theta \log \pi_\theta(a|s)
$$

que proviene de:

$$
\nabla_\theta r(\theta)
= r(\theta)\, \nabla_\theta \log \pi_\theta(a|s)
$$

Este caso permite actualizar la política normalmente.

---

##### 🔹 2. Cuando el ratio se sale del rango (clipping activado)

Si:

$$
r(\theta) < 1 - \epsilon 
\quad \text{o} \quad 
r(\theta) > 1 + \epsilon
$$

el objetivo usa la versión recortada:

$$
\text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) A
$$

Esta expresión es **constante respecto a** $\theta$, por lo que:

$$
\nabla_\theta L = 0
$$

➡ No se actualiza la política para esta transición.  
➡ Se evita que el paso de actualización sea demasiado grande.


De esta forma, PPO evita que la política cambie demasiado rápido y se vuelva inestable.

---

#### Actualización Actor–Critic en PPO

PPO sigue el esquema Actor–Critic:

- **Critic ($V_\phi$)** estima $V(s)$ para construir la función Advantage:  
  $$A = R(\tau) - V_\phi(s)$$

- **Actor ($\pi_\theta$)** actualiza sus parámetros usando el objetivo clipped.

El gradiente real se calcula como:

$$
\nabla_\theta J(\pi_\theta) =
\mathbb{E}_\tau
\left[
\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A_t
\right]
$$

pero modulado por el clipping, es decir, sólo se propaga si el ratio está dentro del rango permitido.

Esto produce:
- actualización para la acción elegida:  

$$
\nabla_\theta J(\pi_\theta)
= \mathbb{E}_{\tau \sim \pi_\theta}
\left[
\sum_{t=0}^{T}
\frac{s_i (1 - s_i)}{sOld_j} \cdot
\text{Adv}
\right]
$$

- actualización ligera o nula para las acciones no elegidas:  

$$
\nabla_\theta J(\pi_\theta)
= \mathbb{E}_{\tau \sim \pi_\theta}
\left[
\sum_{t=0}^{T}
\frac{s_i (s_j)}{sOld_j} \cdot
\text{Adv}
\right]
$$

según si fueron penalizadas por el clipping.

En PPO, la actualización de parámetros depende de cómo cambian las probabilidades de la política nueva respecto a la antigua: 

> Para la acción elegida, el gradiente es fuerte: aumenta o disminuye su probabilidad según el Advantage, y se escala por el ratio entre política nueva y vieja. Esto refuerza buenas acciones y penaliza malas, pero solo dentro de un límite seguro determinado por el clipping.

> Para las acciones no elegidas, la actualización es indirecta y mucho más pequeña: se ajustan sus probabilidades para mantener una distribución coherente, pero sin alterar drásticamente la política. Si el ratio sale del rango permitido, PPO aplica clipping y ambos tipos de actualizaciones se reducen o eliminan, evitando cambios bruscos o inestables.

**En esencia: la acción elegida recibe el update principal, las no elegidas solo pequeños ajustes, y PPO asegura que nada cambie demasiado rápido.**

---

#### Ventajas de PPO
- Reduce varianza y evita actualizaciones peligrosas.
- Mantiene estabilidad similar a TRPO sin su costo computacional.
- Permite múltiples pasos de optimización por cada batch (a diferencia de PG clásico).
- Es actualmente uno de los métodos estándar en entornos de RL modernos.

---

### Comparación PPO y TRPO
| Método | Estabilidad | Coste computacional | Control del cambio |
|--------|-------------|---------------------|--------------------|
| Policy Gradient | Baja | Bajo | Ninguno |
| **TRPO** | Muy alta | **Muy alto** | KL Divergence estricta |
| **PPO** | Alta | Bajo/Medio | Ratio con clipping (trust region suave) |

---

## **Deep Q-Networks (DQN)**

Cuando el espacio de estados es demasiado grande para usar una tabla Q clásica (como en videojuegos tipo Atari donde los estados son **imágenes**), se utiliza una **Red Neuronal Convolucional (CNN)** para aproximar la función de acción-valor:

$$
Q_\theta(s, a)
$$

### **Idea General**
- **Input:** Una imagen o stack de imágenes (estado).
- **Output:** Un vector de valores Q, uno por cada acción posible.
- **Aprendizaje:** La red no aprende a “clasificar”, sino a detectar **características visuales que indican valor futuro**.



## **¿Qué aprende realmente un DQN?**
Una CNN de visión tradicional aprende *features* para reconocer objetos.  
Un **DQN** aprende *features* que le dicen:

> "Si estás viendo este patrón visual, esta acción futura tiende a generar buena recompensa."

Es decir, aprende a ver la pantalla como un humano experto: detecta **señales útiles para sobrevivir, esquivar, atacar, etc.**



### **El Ciclo Completo de Aprendizaje de un DQN**

---

### **1. Observación del Estado**

El agente recibe el estado actual:

$$ s_t $$

que suele ser una imagen o un stack de varios frames (para capturar movimiento).

La red neuronal (CNN + capas densas) produce un **vector de valores Q** para cada acción:

$$
Q_\theta(s_t, a_1),\;
Q_\theta(s_t, a_2),\;
\dots,\;
Q_\theta(s_t, a_n)
$$

Cada valor representa la estimación de cuán buena es cada acción desde ese estado.

---

### **2. Selección de Acción (Exploración vs. Explotación)**

El agente decide la acción mediante una política **$\varepsilon$-greedy**:

- Con prob. $\varepsilon$: toma una **acción aleatoria** (explora).  
- Con prob. $1 - \varepsilon$: toma la **mejor acción según la red**.

Formalmente:

$$
a_t = 
\begin{cases}
\text{acción aleatoria}, & \text{si } \text{Uniform}(0,1) < \varepsilon \\
\arg\max_a Q_\theta(s_t, a), & \text{si } \text{Uniform}(0,1) \ge \varepsilon
\end{cases}
$$

---

### **3. Ejecución de la Acción**

Tras ejecutar $a_t$, el entorno devuelve:

- el siguiente estado $s_{t+1}$
- la recompensa inmediata $r_t$
- un indicador $done$ que dice si el episodio terminó

---

### **4. Almacenamiento en Replay Buffer**

Se guarda la transición completa:

$$
(s_t, a_t, r_t, s_{t+1}, done)
$$

El **Replay Buffer** permite:

- romper correlaciones temporales en los datos  
- entrenar la red con *mini-batches* independientes  
- reutilizar experiencias muchas veces

---

### **5. Muestreo de un Mini-Batch**

Para entrenar, se selecciona un conjunto aleatorio de experiencias:

$$
\{(s_i, a_i, r_i, s'_i, done_i)\}_{i=1}^N
$$

Esto da gradientes más estables que entrenar con datos consecutivos del episodio.

---

### **6. Cálculo del Target TD**

El objetivo (target) para el aprendizaje proviene del método de **Temporal Difference (TD)** y se calcula usando la **Target Network** $Q_{\theta^-}$, una copia estable de la red.

- Si la transición **no es terminal**:

$$
y_i = r_i + \gamma \max_{a'} Q_{\theta^-}(s'_i, a')
$$

- Si **es terminal**:

$$
y_i = r_i
$$

La red principal NUNCA se usa para el $\max$ aquí; por estabilidad, solo se usa la Target Network.

---

### **7. Actualización de la Q-Network**

La red principal se entrena minimizando el error cuadrático entre la predicción y el target:

$$
L(\theta) = 
\frac{1}{N} \sum_{i=1}^N 
\left( y_i - Q_\theta(s_i, a_i) \right)^2
$$

Actualización por gradiente descendente:

$$
\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)
$$

---

### **8. Actualización de la Target Network**

Cada cierto número de pasos, sincronizamos las redes:

- **Hard update**:

$$
\theta^- \leftarrow \theta
$$

- **Soft update** (más estable):

$$
\theta^- \leftarrow \tau \theta + (1 - \tau)\theta^-
$$

donde $0 < \tau \ll 1$ (ej. $10^{-3}$).

---


### **Resumen Conceptual**
**Un DQN:**
- Observa un estado visual.  
- Estima $Q(s, a)$ mediante una CNN.  
- Usa TD-learning para ajustar esos valores.  
- Aprende a reconocer patrones visuales que indican qué acciones son mejor a largo plazo.  
- Se entrena de manera estable gracias a **Replay Memory** y **Target Networks**.

---

## Model-Based & Advanced Architectures
El aprendizaje por refuerzo basado en modelos entrena un modelo a partir del muestreo de la dinámica del entorno y entrena su política a partir del muestreo de este modelo. A continuación se presentan estos modelos:

---

### **World Models**

Los *World Models* buscan que el agente no solo reaccione al entorno, sino que **aprenda su propia simulación interna del mundo**. La gracia es que el agente deja de entrenar directamente sobre imágenes crudas y empieza a entrenar en un espacio más simple y estructurado. Este “mundo interno” se construye usando tres módulos: un **VAE**, un **MDN-RNN** y un **Controller**.

#### **¿Qué es el Espacio Latente?**

Un **espacio latente** es una **versión comprimida de los datos originales** donde solo se conservan las características más importantes.  
Este espacio permite representar información compleja en pocas dimensiones, facilitando el análisis, la predicción y la generación de datos.

En muchos modelos, este espacio se describe mediante los **parámetros de varias distribuciones gaussianas** (medias y varianzas), lo que permite capturar estructuras complejas del mundo real en una forma compacta y manipulable.

Ejemplo: Una imagen de 64×64×3 son 12 288 valores. El VAE puede convertir eso en un vector de, digamos, 32 dimensiones.

Lo importante:

- No es un “pixelado” ni un recorte; es una **codificación abstracta** de las características relevantes.  
- En un buen espacio latente, puntos cercanos representan estados visualmente o semánticamente parecidos.  
- Aprender políticas en este espacio es más fácil porque el agente trabaja con una **versión organizada** del mundo en lugar de imágenes ruidosas y enormes.

En resumen: el latente es “lo que necesitas saber, sin la basura”.


#### **1. VAE (Vision Module)**

El **Variational Autoencoder** toma cada frame del entorno y lo comprime a un vector latente:

$$ z_t = \text{Encoder}(s_t) $$

Ese vector:

- contiene la información relevante de la imagen  
- elimina detalles irrelevantes  
- sigue una distribución gaussiana aprendida

La estructura general es:

$$
s_t \rightarrow \text{Encoder} \rightarrow z_t
$$

y también aprende a decodificar:

$$
z_t \rightarrow \text{Decoder} \rightarrow \hat{s}_t
$$

Esto obliga al VAE a aprender una representación compacta y útil.


#### **2. MDN-RNN (Memory Module)**

Una vez que tenemos estados comprimidos $z_t$, el siguiente paso es aprender **cómo evoluciona el mundo**.

Para eso, se usa un **Recurrent Neural Network** (típicamente un LSTM), pero no cualquiera: se convierte en un **Mixture Density Network** (MDN).  Esto significa que la red no predice un único siguiente estado, sino los **parámetros de varias distribuciones gaussianas**:

$$
P(z_{t+1} \mid z_t, a_t, h_t)
$$

donde $h_t$ es el hidden state del RNN.

¿Por qué mezclar gaussianas?

- Porque el futuro no siempre es determinista.  
- En muchos juegos, desde un mismo estado pueden ocurrir varios eventos posibles.  
- El MDN captura esa **incertidumbre** al predecir múltiples gausianas (cada una con su media, varianza y peso).

Lo que hace el MDN-RNN:

- Modela las dinámicas del entorno en el espacio latente.  
- Predice si el episodio terminará pronto.  
- Mantiene una memoria $h_t$ que representa el “estado interno” de la secuencia.

En fórmula simplificada:

$$
(z_t, a_t, h_t) \rightarrow \text{MDN-RNN} \rightarrow (\text{Gaussians for } z_{t+1},\; \text{done probability})
$$

#### **3. Controller (Policy Module)**

Ahora que ya existe:

- una representación visual comprimida $z_t$
- un modelo del futuro $h_t$

solo falta un módulo que tome decisiones.

El **Controller** suele ser una red neuronal muy pequeña (incluso lineal en el paper original):

$$
a_t = C(z_t, h_t)
$$

Este módulo es el que implementa la política. Lo interesante:

- Ya no necesita ver imágenes crudas.  
- Ya no necesita aprender las dinámicas del entorno.  
- Solo aprende a actuar usando el simulador interno que construyeron el VAE y el MDN-RNN.

Esto reduce brutalmente la complejidad del problema.

---

#### **La Gran Ventaja: El Agente Puede “Soñar”**

Como el MDN-RNN puede predecir $z_{t+1}$ y el indicador de finalización, el agente puede:

- **simular episodios completos dentro de su mente**,  
- sin tocar el entorno real,  
- generando millones de experiencias baratas y rápidas.

El Controller puede entrenarse completamente dentro de esta simulación:

$$
\text{Controller} \;\text{entrena en el mundo generado por}\; (VAE + MDN\text{-}RNN)
$$

y luego se transfiere al entorno real.

---

#### **Pros y Contras**

**Pros:**
- Enorme eficiencia de muestras: la política se entrena en el espacio latente y en simulaciones internas.  
- Permite entrenar múltiples tareas sobre el mismo modelo del mundo.  
- El VAE reduce la complejidad del input visual de forma masiva.

**Contras:**
- El MDN-RNN a veces falla al modelar dinámicas difíciles.  
- El Controller puede aprender a explotar “bugs” del mundo simulado. Luego esas políticas no funcionan en el entorno real.  
- Requiere entrenar tres modelos separados (cierta complejidad).

---

### **Deep Planning Network (PlaNet)**

PlaNet es una mejora sobre **World Models**, diseñada para realizar *planeación* directamente en el **espacio latente**, sin necesidad de entrenar una política tradicional.

#### 🔹 **Idea Central**
En lugar de predecir imágenes o entrenar un policy network, PlaNet:

1. **Aprende un modelo del mundo en espacio latente.**  
   El modelo predice:
   - el siguiente estado latente,
   - la recompensa futura,
   - y la probabilidad de terminar el episodio.

2. **Simula (“rollouts”) miles de trayectorias posibles dentro del modelo**, sin usar el entorno real.

3. **Elige la secuencia de acciones** que maximiza la suma de recompensas simuladas.

> Es decir: *planifica*, no solo *reacciona*.

#### 🔹 ¿Cómo funciona PlaNet?

##### **1. Aprendizaje del modelo dinámico**
El sistema entrena un modelo en espacio latente que captura:
- transiciones latentes:  
  $$ z_{t+1} = f(z_t, a_t) $$
- recompensas:  
  $$ r_t = g(z_t, a_t) $$
- terminación del episodio.

Este modelo NO trabaja con imágenes directamente; utiliza una codificación latente compacta.

##### **2. Rollouts imaginados**
Con el modelo entrenado, PlaNet **simula miles de futuros posibles**:

$$
(z_t, a_t) \rightarrow z_{t+1} \rightarrow z_{t+2} \rightarrow \dots
$$

Cada secuencia genera una recompensa acumulada:

$$
R = \sum_{k=0}^{H} r_{t+k}
$$

donde $H$ es el horizonte de planeación.


##### **3. Optimización de la secuencia de acciones**
PlaNet usa métodos como **CEM (Cross-Entropy Method)** para buscar acciones que maximicen $R$.

No aprende una política explícita:  
> *elige acciones optimizando directamente la recompensa predicha*.

---
#### ✔️ **Pros**
- **>5000% más eficiente en muestras**: casi no requiere interacción con el entorno real.
- Planea en un espacio comprimido → es más rápido y estable que trabajar con imágenes.
- Puede reutilizar el mismo modelo para múltiples tareas (*multi-task*).

#### ✖️ **Contras**
- El **modelo del mundo puede fallar** al representar dinámicas complejas.
- El agente puede aprender a **explotar errores del modelo**, generando políticas que no funcionan en el entorno real.
- Requiere cómputo considerable para simular miles de trayectorias en cada paso.

#### **Resumen en una frase**
PlaNet no aprende una política:  
> **“Imagina” miles de futuros en su espacio latente, evalúa sus recompensas y actúa siguiendo el mejor plan.**

---

### **Curiosity Driven Exploration**

Cuando un entorno tiene **recompensas extremadamente escasas**, el agente puede pasar miles de episodios sin recibir señal útil.  
Para evitar que la política “no aprenda nada”, se introduce un mecanismo interno de motivación:

#### **🔹 Intrinsic Curiosity Module (ICM)**

El ICM genera una **recompensa intrínseca (señal de motivación generada internamente por el agente, no por el entorno, cuyo propósito es incentivar la exploración)** que motiva al agente a explorar zonas donde el modelo aún no sabe predecir bien.

El módulo tiene **dos componentes principales**:

##### **1. Inverse Model (IM)**  
Recibe dos estados consecutivos en *feature space*:

$$ \phi(s_t),\; \phi(s_{t+1}) $$

y predice qué acción los conectó:

$$ \hat{a}_t = IM(\phi(s_t), \phi(s_{t+1})) $$

Su pérdida es una **cross-entropy**:

$$
L_I = \text{CE}(a_t,\; \hat{a}_t)
$$

Sirve para aprender representaciones de estados útiles y consistentes (evita triviales cambios de píxel).

---

##### **2. Forward Model (FM)**  
Predice el siguiente estado latente usando el estado actual y la acción real:

$$
\hat{\phi}(s_{t+1}) = FM(\phi(s_t), a_t)
$$

La **intrinsic reward** proviene del error de esta predicción:

$$
r^{int}_t = \frac{1}{2} \left\|\, \hat{\phi}(s_{t+1}) - \phi(s_{t+1}) \,\right\|^2
$$

Zonas donde el modelo falla → **zonas interesantes** → el agente quiere visitarlas.

---

#### **3. Función de Pérdida Total del ICM (Intrinsic Curiosity Module)**

El ICM genera **recompensa intrínseca** a partir de qué tan difícil es predecir las consecuencias de las propias acciones del agente.  
Para lograrlo, el módulo usa dos partes:

1. **Inverse Model (IM):**  
   Aprende a predecir la acción ejecutada $a_t$ a partir del par de estados codificados:  
   $$
   (\phi(s_t), \phi(s_{t+1}))
   $$  
   Esto obliga al codificador $\phi(\cdot)$ a retener únicamente *información controlable por el agente*.

2. **Forward Model (FM):**  
   Predice la representación futura:  
   $$\hat{\phi}(s_{t+1}) = F(\phi(s_t), a_t)
   $$  
   El error de esta predicción mide qué tan sorprendente o novedoso es el cambio en el entorno, y se usa como **recompensa intrínseca**.

**Pérdida total del ICM**

El ICM entrena ambos modelos mediante la función:

$$
L_{ICM} = (1 - \beta) L_I + \beta L_F
$$

donde:

- $\beta$ controla el equilibrio entre aprender a **predecir acciones** (IM) y **predecir el futuro** (FM).
- $L_I$: pérdida del *inverse model*, generalmente entropía cruzada al predecir $a_t$.
- $L_F$: pérdida del *forward model* en espacio latente:

$$
L_F = \left\| \hat{\phi}(s_{t+1}) - \phi(s_{t+1}) \right\|^2
$$

---

#### **Recompensa intrínseca como error de predicción**

La recompensa intrínseca surge del error del forward model:

$$
r_t^{int} = \frac{1}{2}
\left\| 
\hat{\phi}(s_{t+1}) - \phi(s_{t+1}) 
\right\|^2
$$

Esta cantidad es alta cuando el agente encuentra **situaciones desconocidas o difíciles de predecir**, incentivando la exploración.

---

#### **Recompensa total usada por la Política**

La política sigue optimizando una recompensa combinada:

$$
r_t^{total} = r_t^{extrinsic} + \lambda \, r_t^{int}
$$

donde:

- $r_t^{extrinsic}$ es la recompensa real del entorno,
- $r_t^{int}$ proviene del ICM,
- $\lambda$ controla cuánto "pesa" la curiosidad.

---

#### **Interpretación completa**

Cuando el agente toma una acción $a_t$ en $s_t$ y pasa a $s_{t+1}$:

1. Ambos estados se codifican:  
   $$\phi(s_t),\; \phi(s_{t+1})$$

2. El **Inverse Model** aprende la acción que causó ese cambio → hace que $\phi(\cdot)$ ignore *ruido o elementos incontrolables*.

3. El **Forward Model** predice $\hat{\phi}(s_{t+1})$ a partir de $\phi(s_t)$ y $a_t$.  
   Su error define la curiosidad.

4. La política maximiza la suma de recompensas extrínsecas + intrínsecas.

**Resultado:**  
El agente explora de forma robusta, sin distraerse con cambios aleatorios del entorno que **no están afectados por sus acciones** (un beneficio clave del codificador $\phi$).

- Le da al agente **motivación propia** para explorar, incluso cuando no hay recompensas externas.
- Actúa como **preentrenamiento de exploración**: aprende la estructura del entorno.
- Funciona especialmente bien en **entornos con recompensas muy escasas o retrasadas**.

---

#### **Problemas y Limitaciones**

##### **1. Exploitation de ruido**  
Si hay una zona del entorno donde las observaciones cambian aleatoriamente, el Forward Model fallará siempre →  
intrinsic reward muy alto → el agente entra en un *loop* buscando solo esa zona.

##### **2. Aprendizajes no transferibles**  
El agente puede aprender a explorar bien, pero no necesariamente a optimizar la tarea si:
- el objetivo real está lejos,  
- el entorno tiene dinámicas engañosas.

##### **3. Coste adicional**  
Entrenar IM + FM + Policy agrega complejidad computacional.

---

#### **Resumen**

> **Curiosity Driven Exploration** permite que el agente encuentre comportamientos útiles incluso sin recompensas externas.  
El ICM aprende qué partes del entorno son *difíciles de predecir* y usa ese error como una recompensa interna para impulsar la exploración, aunque puede ser engañado por entornos ruidosos.

---
### Meta-Reinforcement Learning (Meta-RL / Learning to Learn)

Meta-RL busca que un agente **aprenda a adaptarse rápido** cuando la tarea o las reglas del entorno cambian (p. ej. bandits con probabilidades cambiantes, metas que se mueven en un laberinto, variantes de un videojuego). En vez de aprender una sola política para un problema fijo, se aprende una **estrategia meta** que permite obtener buenas políticas con muy pocos pasos de interacción en una nueva tarea.

> **¿Qué problema resuelve?**
En entornos no estacionarios o en familias de tareas (distribución de tareas $p(\mathcal{T})$), los métodos RL tradicionales (model-free) requieren muchas interacciones para reaprender. Meta-RL intenta **aprender la estructura** entre tareas para que la adaptación sea *rápida* (few-shot).

Se define una distribución de tareas $p(\mathcal{T})$. Para cada tarea $\mathcal{T}$ tenemos una pérdida/función de rendimiento $L_{\mathcal{T}}(\cdot)$. El objetivo meta es:

$$
\min_\theta \; \mathbb{E}_{\mathcal{T}\sim p(\mathcal{T})}\big[\, L_{\mathcal{T}}\big(U(\theta, \mathcal{D}_{\mathcal{T}})\big)\,\big]
$$

donde:
- $\theta$ son los parámetros meta (inicialización, arquitect., etc.).  
- $\mathcal{D}_{\mathcal{T}}$ son las pocas experiencias recolectadas en la tarea $\mathcal{T}$.  
- $U(\theta,\mathcal{D}_{\mathcal{T}})$ es la **regla de adaptación** (por ejemplo una actualización de gradiente, o la evolución del estado oculto en una RNN).

---

#### Estrategias comunes en Meta-RL

##### 1. **Optimization-based (p. ej. MAML for RL)**
- Aprender una **inicialización** de parámetros tal que unas pocas actualizaciones de gradiente en una nueva tarea producen una buena política.

##### 2. **Recurrent / Contextual policies (p. ej. RL², Prefrontal Network)**
- La política incluye memoria (LSTM/GRU/Transformer). En lugar de actualizar parámetros con gradiente, la **memoria interna** (estado oculto) se actualiza automáticamente con secuencia de $(s,a,r)$ y codifica el *contexto* / estructura de la tarea.
- Entrenamiento meta: expones la RNN a múltiples episodios por tarea; la RNN aprende a “leer” señales (acciones, recompensas anteriores) y adaptar comportamiento *on-the-fly*.

##### 3. **Model-based meta-RL**
- Aprender un **modelo de dinámica** que sea compartible entre tareas y usarlo para planificar o para adaptar política rápidamente.
- Ejemplos: adaptar parámetros del modelo con pocas muestras y planear en ese modelo adaptado.

---

#### **Arquitecturas Prefrontal Network**
- **Entrada:** además del estado actual $s_t$, se alimentan señales de contexto: acción previa $a_{t-1}$, recompensa previa $r_{t-1}$, bandera de fin de episodio, otros indicadores.  
- **Red recurrente (LSTM/GRU):** mantiene un estado oculto $h_t$ que resume historia breve y permite inferir la tarea actual.  
- **Salida:** política $\pi(a|s,h)$ y/o critic $V(s,h)$.  
- El LSTM actúa como una *memoria de meta-aprendizaje* (simula la función de la corteza prefrontal).

---

#### Ejemplo intuitivo: 2-armed bandit no estacionario
- Tarea: cada episodio, las probabilidades de los dos brazos pueden cambiar.
- Un agente meta-entrenado aprende a usar la secuencia de recompensas recientes para inferir cuál brazo es mejor (sin fine-tune), gracias a su memoria (LSTM) o a una inicialización que se adapta rápido (MAML).

---

#### ¿Por qué funciona? — Intuición
- Muchas tareas comparten estructura (p. ej. “hay dos tipos de dinámica”, “las recompensas cambian lentamente”).  
- Meta-RL explota esas regularidades: aprende **cómo aprender** — reglas de actualización o políticas recurrentes que codifican estrategias de exploración y explotación eficientes bajo incertidumbre.

---

### **Decision Transformer (DT)**  
Un enfoque moderno que reformula el *Reinforcement Learning* como un **problema de modelado de secuencias**, de manera similar a GPT o modelos tipo BERT.

En lugar de aprender valores, TD-errors o políticas explícitas, el Decision Transformer aprende a **predecir la siguiente acción** a partir de una secuencia pasada de:
- estados
- acciones
- recompensas
- *return-to-go* (suma futura de recompensas deseada)

DT aprende patrones de **trayectorias completas** usando *causal attention*.

---

#### **Cómo funciona**

##### **1. Representación como secuencia**
Cada paso se convierte en tokens:

$$
(R_t, s_t, a_t), (R_{t+1}, s_{t+1}, a_{t+1}), \dots
$$

donde:

- $R_t$ = *return-to-go* (recompensa futura que queremos alcanzar)  
- $s_t$ = estado  
- $a_t$ = acción  

El modelo recibe varios de estos tokens concatenados como **una sola secuencia**, igual que una oración en NLP.

---

##### **2. Modelo Transformer**
Como en GPT, utiliza *masked causal attention*:

- solo ve el pasado  
- predice el siguiente token: **la acción óptima**  

El mecanismo de atención permite:
- reconocer patrones largos en secuencias  
- ignorar partes irrelevantes del estado  
- aprender trayectorias buenas incluso cuando la mayoría son subóptimas

---

##### **3. Predicción de la acción**
El transformador se entrena para resolver:

$$
(a_{t}) = f_{\text{Transformer}}(R_{t}, s_{t}, a_{t-1}, s_{t-1}, \dots)
$$

Es decir, aprende qué acción llevaría al *return-to-go* deseado.

---

##### **4. Ventajas clave**
- No usa TD-learning ni una función valor.  
- Maneja datos **offline**: aprende solo de secuencias grabadas.  
- Aprovecha *attention* para reconstruir políticas de alto rendimiento aunque los datos vengan de comportamientos no óptimos.  
- Escala muy bien con datos masivos, igual que los modelos de lenguaje.

---

##### **5. Intuición**
DT aprende:  
> *“En secuencias donde el objetivo era alto, la gente que hizo esto tomó estas acciones… así que yo también las tomaré.”*

---

## 10. RL from Human Feedback (RLHF)

Esquema para alinear IAs (como ChatGPT) con intención humana:
1.  **Pretraining:** Supervised Learning (predicción de next-token).
2.  **SFT (Supervised Fine-Tuning):** Se ajusta con ejemplos de buenas preguntas/respuestas humanas.
3.  **Reward Modeling:** Se entrena una red neuronal para predecir una puntuación (score) basada en rankings hechos por humanos (esto es mejor que aquello).
4.  **RL Optimization (PPO):** Se usa PPO para optimizar el modelo de lenguaje usando el Reward Model como fuente de recompensa.


---

# Information Theory 

## 1. ¿Qué es la Información?
La **información** se entiende como *variaciones en los datos*.  
Un **mensaje** es un conjunto de variaciones estructuradas siguiendo un patrón.

El objetivo central de la teoría de la información es:
> **Encontrar el mejor patrón para transmitir mensajes por un canal ruidoso minimizando incertidumbre y pérdida.**

---

## 2. Transmitiendo un Mensaje en un Canal Ruidoso
Cuando enviamos un mensaje (por ejemplo, una palabra):

- El canal puede **corromper o perder** partes del mensaje.  
- El receptor debe pedir aclaraciones (“¿qué letra era?”).
- Para minimizar preguntas, debemos diseñar un sistema eficiente para identificar el símbolo enviado.

---

## 3. Reduciendo al Mínimo las Preguntas (Bits)
La estrategia óptima es codificar mensajes usando **dos símbolos**:  
`0` y `1` → mínima cantidad de estados.

Cada pregunta del receptor divide el espacio de posibilidades en dos:
- “¿Está en la primera mitad?”
- “¿Está en la segunda mitad?”

Ejemplo para letras A–Z:
- 26 símbolos → ¿cuántas preguntas mínimas?  
- Resolver:  
  $$
  2^x = 26 \quad \Rightarrow \quad x = \log_2 26 ≈ 4.7 \text{ bits}
  $$

Ejemplo para un mazo de 52 cartas:
$$
x = \log_2 52 ≈ 5.7 \text{ bits}
$$

---

## 4. Información Total de un Mensaje

La **información total** de un mensaje depende de:

- $n$: número de símbolos a transmitir  
- $s$: número de símbolos posibles (tamaño del alfabeto o conjunto de opciones)

La fórmula para calcular la información total en **bits** es:

$$
I = n \cdot \log_2(s)
$$

### Ejemplos:

1. **Transmisión de letras:**
   - Alfabeto de 26 letras
   - Mensaje de 6 letras
   - Cada letra requiere en promedio:
     
     $$
     \log_2(26) \approx 4.7 \text{ bits}
     $$
     
   - Información total del mensaje:
     
     $$
     I = 6 \cdot 4.7 \approx 28.2 \text{ bits}
     $$

2. **Transmisión de cartas de un mazo:**
   - Mazo de 52 cartas
   - Mensaje de 5 cartas
   - Cada carta requiere en promedio:
     
     $$
     \log_2(52) \approx 5.7 \text{ bits}
     $$
     
   - Información total del mensaje:
     
     $$
     I = 5 \cdot 5.7 \approx 28.5 \text{ bits}
     $$

> **Interpretación:** Cuantos más símbolos posibles tenga el conjunto ($s$), más bits se necesitan para transmitir cada símbolo. Esta medida refleja la **incertidumbre** de cada mensaje antes de ser transmitido.


---

## 5. Información con Probabilidades Desiguales

Cuando los símbolos no son igualmente probables, debemos calcular la **información promedio** usando **valores esperados**.

---

### Ejemplo: 4 símbolos con probabilidades distintas

Supongamos:

$$
P(A) = 0.5, \quad P(B) = 0.125, \quad P(C) = 0.125, \quad P(D) = 0.25
$$

Para identificar un símbolo, el número de preguntas se ajusta según la probabilidad:

1. Primero se pregunta si es $A$:  
   - Probabilidad = 0.5 → **1 pregunta**
2. Luego $D$:  
   - Probabilidad = 0.25 → **2 preguntas**
3. Finalmente $B$ y $C$:  
   - Probabilidad = 0.125 cada uno → **3 preguntas**  

---

### Cálculo del número esperado de preguntas

$$
\#\text{questions} = P(A) \cdot 1 + P(D) \cdot 2 + P(B) \cdot 3 + P(C) \cdot 3
$$

$$
\#\text{questions} = 0.5 \cdot 1 + 0.25 \cdot 2 + 0.125 \cdot 3 + 0.125 \cdot 3 = 1.75
$$

> Esto representa el **número promedio de preguntas** necesarias para identificar un símbolo en este conjunto no uniforme.

---

### Entropía

La **entropía** $H$ mide la incertidumbre promedio de la distribución:

$$
H = - \sum_i p_i \log_2(p_i)
$$

Aplicando al ejemplo:

$$
H = - [0.5 \log_2 0.5 + 0.25 \log_2 0.25 + 0.125 \log_2 0.125 + 0.125 \log_2 0.125] \approx 1.75 \text{ bits}
$$

**Interpretación:**

- Mayor probabilidad de un símbolo → menos preguntas necesarias.  
- Menor entropía $H$ → menor incertidumbre del mensaje.  
- La entropía refleja **la cantidad promedio de información** necesaria para transmitir un mensaje considerando probabilidades desiguales.


---

## 6. Entropía
La **entropía** mide cuánta incertidumbre existe en una distribución:

$$
H = - \sum_{i} p_i \log_2 p_i
$$

Interpretación:
- **H alta** → alta incertidumbre, símbolos equiprobables.  
- **H baja** → hay símbolos mucho más probables que otros.

Ejemplos:
- Distribución de sexo 30–34 años: $H = 0.99$
- Distribución en militares (91% hombres): $H = 0.43$

---

## 7. Information Gain (Ganancia de Información)
Es la reducción de entropía al conocer un atributo:

$$
IG = H(D) - H(D|a)
$$

Ejemplo:
$$
0.99 - 0.94 = 0.05
$$

Muy utilizado en:
- Árboles de decisión
- Selección de atributos

---

## 8. Divergencia KL (Kullback–Leibler)
Mide cuán diferente es una distribución **P** de una distribución **Q**:

$$
KL(P‖Q) = \sum_i P(x_i) \log \frac{P(x_i)}{Q(x_i)}
$$

Propiedad clave:
- **No es simétrica:**  
  $$
  KL(P‖Q) \neq KL(Q‖P)
  $$

---

## 9. Cross-Entropy (Pérdida en Redes Neuronales)
Usada en clasificación y modelos probabilísticos, otro método para calcular la diferencia entre dos distribuciones:

$$
H(P,Q) = - \sum_i P(x_i)\log Q(x_i)
$$

Interpretación:
- P = la *verdad* (dataset)
- Q = nuestro *modelo*
- Cross-Entropy dice cuán mal Q aproxima a P.

---

## 10. Relación entre Verdad, Datos y Modelo
En Machine Learning:

$$
P(modelo) \approx P(datos) \approx P(verdad)
$$

- La **verdad** es incognoscible.  
- Los **datos** son nuestro proxy.  
- El **modelo** intenta aproximarlos.

---


