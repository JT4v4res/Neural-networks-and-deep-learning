# Neural Network

## Português

Como visto anteriormente, um perceptron de camada única não é capaz de resolver o problema do OU exclusivo (XOR)
dessa forma, tentando resolver esse problema, os pesquisadores idealizaram um modelo bio-inspirado para tal, e se conectarmos
diversos neurônios em camadas de forma que a saída de um alimente diversos outros? É assim que surgem as Redes Neurais Artificiais(RNAs).

A base de uma Rede Neural consiste em basicamente conectar diversos neurônios em camadas diferentes, de forma que a rede
consiga aprender um espaço não linear e combinar diversas entradas de forma a encontrar um hiperplano que melhor separe as características,
o conhecimento que é representado na rede são os pesos, de forma que a rede tenta entender qual a melhor combinação de pesos
que minimiza uma determinada função de perda.

A Rede Neural possui uma configuração muito interessante, ela é subdividida em camadas,
sendo divididade em camada de entrada, camada de saída e diversas camadas intermediáras, que são as que se situam entre a entrada e a saída.

![Exemplo de rede neural](https://7793103.fs1.hubspotusercontent-na1.net/hubfs/7793103/Imported_Blog_Media/unnamed-6-1.jpg)

### Camada de entrada

A camada de entrada possui uma quantidade de neurônios igual ao número de posições no vetor de entrada,
seu único papel é repassar os valores de entrada para todos os neurônios da primeira camada intermediária.

### Camadas intermediárias

As camadas intermediárias, por sua vez, realizam diversos cálculos em seus neurônios e os enviam para a próxima camada, até
atingirem a camada de saída.

### Camada de saída

A camada de saída possui a responsabilidade de dizer qual a classificação final de uma determinada amostra que a rede recebe.

## Aprendizagem de uma rede

Para realizar o aprendizado, a rede precisa propagar a entrada entre todas as suas camadas ocultas até atingir a saída,
ao atingir a saída, precisamos calcular o erro para que possamos atualizar os pesos baseados nesse erro, assim como no Perceptron.

No entanto, aqui temos alguns problemas pois como iremos atualizar os pesos da camada oculta baseados no erro de saída?
Para isso, utilizamos um algoritmo poderoso chamado de Backpropagation, que é responsável por atualizar os pesos baseado
nas derivadas parciais, a regra delta.

Vamos entender um pouco melhor esse funcionamento.

### Propagação direta

Na propagação direta, ou forward propagation, a rede recebe as entradas e realiza a passagem das mesmas através das camadas da rede,
cada neurônio recebe as ativações dos neurônios anteriores e realiza o mesmo procedimento de combinação linear do perceptron
para essas ativações, devolvendo uma ativação também, de modo que ao final da iteração, a ativação da camada de saída definirá nossa classificação final.

![Feed-Forward](https://private-user-images.githubusercontent.com/52336334/286694015-b4a0c665-b67d-4e8e-adb8-33e99d97d253.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDk1NTY0OTIsIm5iZiI6MTcwOTU1NjE5MiwicGF0aCI6Ii81MjMzNjMzNC8yODY2OTQwMTUtYjRhMGM2NjUtYjY3ZC00ZThlLWFkYjgtMzNlOTlkOTdkMjUzLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAzMDQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMzA0VDEyNDMxMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTJlZTA2NDk5MjhjMWI2Zjk1ZTY4MzhiYzllMjA2NDJjMjI5Njc5MGQ2MmJmYTk1ZGVmMjEzNGNjOTVlZDg4NWQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.8ops4fis2lbqlrNbPqz-aOzFl6H38Yb6M7FvlwDGLtQ)

Ao final dessa etapa calculamos o erro da nossa saída predita com a saída esperada para podermos atualizar os pesos.

### Propagação reversa

Na propagação reversa, ou backward propagation, nós utilizamos a regra delta para atualizar os pesos camada por camada, como fazemos isso?
Utilizamos o Gradiente Descendente e a regra delta para a realização dessa etapa.

![Backpropagation](https://private-user-images.githubusercontent.com/52336334/286693588-723fab30-0832-4dbc-bba9-3394fe75c0d1.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDk1NTczMTMsIm5iZiI6MTcwOTU1NzAxMywicGF0aCI6Ii81MjMzNjMzNC8yODY2OTM1ODgtNzIzZmFiMzAtMDgzMi00ZGJjLWJiYTktMzM5NGZlNzVjMGQxLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAzMDQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMzA0VDEyNTY1M1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWRlZTQyODlhNjE3OWViNmUzODM4YjkyY2QzZTEwNWIxMzJmMDA4N2VmMjY0Y2ZlZjM5MjVkNDlmODdmYjA0NDQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.Y7SHMQ1n5zvGhHLvoU9f_07e-Wc8UoRYxVrtrzpTGlM)

O Gradiente é responsável por nos indicar a direção na qual nossa função cresce com maior intensidade, ao utilizarmos o gradiente descendente estamos buscando
a região em que nossa função tem o menor crescimento, ou seja, uma região de mínimo local.

![Gradiente descendente](https://upload.wikimedia.org/wikipedia/commons/a/a3/Gradient_descent.gif)

Como queremos minimizar a nossa perda, o gradiente pode nos indicar a região na superfície de perda que obtemos a menor perda.

Já a regra delta consiste em, a partir da derivada da função de perda, utilizamos uma sequência de derivadas para realizar os cálculos,
por exemplo, para a atualização de pesos da penúltima camada teríamos: 
```
Derivada da função de perda * Derivada da função de ativação da penúltima camada.
```

Supondo que nossa última camada seja H, e a camada anterior seja H-1, temos algo como:

```
dH * dH-1 * dH-2 ...
```

Essa seria a fórmula de atualização dos nossos pesos com a backpropagation, dessa forma nossa rede aprende e conseguimos minimizar o erro.

A fórmula geral para atualização do erro na camada é:

```
w = w - α ∇w
```

E dessa forma, nossa rede é capaz de aprender espaços não lineares.

### Exemplos

#### Problema do XOR

Resolução com a nossa rede:

![XOR](Problema%20do%20XOR.png)

#### Classificação binária em espiral

Resolução com a nossa rede:

![Binary spiral](Classificação%20binária%20em%20espiral.png)

#### Regressão quadrática

Resolução com a nossa rede:

![Quadratic regression](Regressão%20quadrática.png)

#### Classificação binária com círculos

Resolução com a nossa rede:

![Binary circles](Classificação%20binária%20com%20círculos.png)

#### Classificação de 5 classes com espiral

Resolução com a nossa rede:

![Multiclass spiral](Classificação%20de%205%20classes%20com%20espiral.png)

## English

As seen previously, a single-layer perceptron is not capable of solving the exclusive OR (XOR) problem.
Thus, trying to solve this problem, researchers devised a bio-inspired model for this, and if we connect
several neurons in layers so that the output of one feeds several others? This is how Artificial Neural Networks (ANNs) emerge.

The basis of a Neural Network consists of basically connecting several neurons in different layers, so that the network
be able to learn a non-linear space and combine several inputs in order to find a hyperplane that best separates the features,
the knowledge that is represented in the network are the weights, so the network tries to understand the best combination of weights
that minimizes a given loss function.

The Neural Network has a very interesting configuration, it is subdivided into layers,
being divided into input layer, output layer and several intermediate layers, which are those located between input and output.

![Exemplo de rede neural](https://7793103.fs1.hubspotusercontent-na1.net/hubfs/7793103/Imported_Blog_Media/unnamed-6-1.jpg)

### Input layer

The input layer has a number of neurons equal to the number of positions in the input vector,
its only role is to pass the input values to all neurons in the first intermediate layer.

### Hidden layers

The intermediate layers, in turn, perform various calculations in their neurons and send them to the next layer, until
reach the output layer.

### Output layer

The output layer is responsible for saying the final classification of a given sample that the network receives.

## Learning from a network

To perform learning, the network needs to propagate the input between all its hidden layers until it reaches the output,
When reaching the output, we need to calculate the error so that we can update the weights based on that error, just like in Perceptron.

However, here we have some problems because how will we update the hidden layer weights based on the output error?
To do this, we use a powerful algorithm called Backpropagation, which is responsible for updating the weights based on
in partial derivatives, the delta rule.

Let's understand this operation a little better.

### Forward propagation

In direct propagation, or forward propagation, the network receives the inputs and passes them through the network layers,
each neuron receives the activations of the previous neurons and performs the same linear combination procedure as the perceptron
for these activations, returning an activation as well, so that at the end of the iteration, the activation of the output layer will define our final classification.

![Feed-Forward](https://private-user-images.githubusercontent.com/52336334/286694015-b4a0c665-b67d-4e8e-adb8-33e99d97d253.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDk1NTY0OTIsIm5iZiI6MTcwOTU1NjE5MiwicGF0aCI6Ii81MjMzNjMzNC8yODY2OTQwMTUtYjRhMGM2NjUtYjY3ZC00ZThlLWFkYjgtMzNlOTlkOTdkMjUzLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAzMDQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMzA0VDEyNDMxMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTJlZTA2NDk5MjhjMWI2Zjk1ZTY4MzhiYzllMjA2NDJjMjI5Njc5MGQ2MmJmYTk1ZGVmMjEzNGNjOTVlZDg4NWQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.8ops4fis2lbqlrNbPqz-aOzFl6H38Yb6M7FvlwDGLtQ)

At the end of this step, we calculate the error of our predicted output with the expected output so that we can update the weights.

### Backward propagation

In reverse propagation, we use the delta rule to update the weights layer by layer, how do we do this?
We use Gradient Descent and the delta rule to carry out this step.

![Backpropagation](https://private-user-images.githubusercontent.com/52336334/286693588-723fab30-0832-4dbc-bba9-3394fe75c0d1.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDk1NTczMTMsIm5iZiI6MTcwOTU1NzAxMywicGF0aCI6Ii81MjMzNjMzNC8yODY2OTM1ODgtNzIzZmFiMzAtMDgzMi00ZGJjLWJiYTktMzM5NGZlNzVjMGQxLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDAzMDQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwMzA0VDEyNTY1M1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWRlZTQyODlhNjE3OWViNmUzODM4YjkyY2QzZTEwNWIxMzJmMDA4N2VmMjY0Y2ZlZjM5MjVkNDlmODdmYjA0NDQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.Y7SHMQ1n5zvGhHLvoU9f_07e-Wc8UoRYxVrtrzpTGlM)

The Gradient is responsible for indicating the direction in which our function grows with greater intensity, when using gradient descent we are looking for
the region in which our function has the lowest growth, that is, a region of local minimum.

![Gradiente descendente](https://upload.wikimedia.org/wikipedia/commons/a/a3/Gradient_descent.gif)

Since we want to minimize our loss, the gradient can tell us the region on the loss surface where we get the lowest loss.

The delta rule consists of, from the derivative of the loss function, we use a sequence of derivatives to perform the calculations,
for example, to update the weights of the penultimate layer we would have:

```
Derivative of the loss function * Derivative of the activation function of the penultimate layer.
```
Assuming our last layer is H, and the previous layer is H-1, we have something like:

```
dH * dH-1 * dH-2 ...
```

This would be the formula for updating our weights with backpropagation, this way our network learns and we can minimize the error.

The general formula for updating the layer error is:

```
w = w - α ∇w
```

And in this way, our network is capable of learning non-linear spaces.

### Examples

#### XOR problem

Resolution with our network:

![XOR](Problema%20do%20XOR.png)

#### Spiral binary classification

Resolution with our network:

![Binary spiral](Classificação%20binária%20em%20espiral.png)

#### Quadratic regression

Resolution with our network:

![Quadratic regression](Regressão%20quadrática.png)

#### Binary classification with circles

Resolution with our network:

![Binary circles](Classificação%20binária%20com%20círculos.png)

#### 5 class classification with spiral

Resolution with our network:

![Multiclass spiral](Classificação%20de%205%20classes%20com%20espiral.png)

