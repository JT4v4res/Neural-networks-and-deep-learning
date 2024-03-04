# Activation functions

## Português

As funções de ativação são parte essencial da rede, são elas que definem a saída geral dos neurõnios e da própria rede.

Cada função de ativação tem suas vantagens e desvantagens, sendo deriváveis o que permite a utilização de aprendizagem por backpropagation.

A escolha de qual função de ativação utilizar vai variar de acordo com o problema que você quer resolver, mas vamos visualizar algumas destas.

### ReLU

A função de ativação ReLU é amplamente utilizada por ser computacionalmente eficiente,
é aplicada numa gama de tarefas, sua vantagem é justamente a eficiência, já que a mesma é 0 para qualquer número negativo,
e retorna o próprio número para entradas >= 0. Uma desvantagem é justamene a questão de retornar 0, o que pode causar o problema
do gradiente desaparecendo, o que prejudica o aprendizado e pode fazer com que determinados neurônios não aprendam devidamente.

![Relu function](https://miro.medium.com/v2/resize:fit:1400/0*17a9Xr_jp1KXlxT8.png)

### Leaky ReLU

Como uma solução para o problema do gradiente desaparecendo, surge a função Leaky ReLU, que contém um vazamento para que os números
negativos não sejam tomados como 0, mas ainda os tornando muito próximos a 0.

![Leaky ReLU](https://pytorch.org/docs/stable/_images/LeakyReLU.png)

### ELU

A ideia da utilização da ELU é a introdução da não-linearidade dentro do nosso modelo, o que pode aumentar sua complexidade,
fazendo com que o mesmo consiga aprender padrões ainda mais complexos, a sua diferença para a ReLU é que os valores negativos
são calculados a partir de exponenciação.

![Elu](https://armandolivares.tech/wp-content/uploads/2022/09/elu-1.png)

### Linear

A função linear, como o próprio nome já diz, é uma reta, cuja sua única função é devolver o mesmo valor que recebe,
sendo aplicável em problemas de classificação multiclasse na camada de saída e de regressão.

![Linear](https://cdn-academy.pressidium.com/academy/wp-content/uploads/2021/12/key-features-of-linear-function-graphs-2.png)


### Sigmoide

A função sigmoide é uma função de ativação utilizada para regressão logística, tem um intervalo de domínio de (0, 1) e faz com que
valores muito grandes fiquem próximos de 1 e valores muito pequenos próximos a 0, normalmente é indicada somente nas camadas de saída
de problemas de classificação binária, pois não possui um bom desempenho computacional nas camadas intermediárias, além de ser muito
propensa ao problema do gradiente desaparecendo.

![Sigmoide](https://miro.medium.com/v2/resize:fit:1400/1*a04iKNbchayCAJ7-0QlesA.png)


### Tanh

A função de ativação tangente hiperbólica é uma função de ativação que segue um comportamento trigonométrico, mas também possui
uma curva similar ao da função sigmoide, com o adicional de permitir valores negativos, além de possuir uma derivada cujos valores não estão
tão próximos de 0 quanto a função sigmoide.

![Tanh](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-05-27_at_4.23.22_PM_dcuMBJl.png)

### Softmax

A softmax é uma função muito utilizada no contexto de classificação multiclasse, onde recebe uma entrada e converte a mesma em uma
distribuição de probabilidade, onde todas as probabilidades somam para 1, é uma gerenalização da regressão logística e é
bastante aplicada sempre que precisamos prever 3 ou mais classes.

![Softmax](https://cdn.botpenguin.com/assets/website/Softmax_Function_07fe934386.png)

## English

Activation functions are an essential part of the network, they define the general output of the neurons and the network itself.

Each activation function has its advantages and disadvantages, being derivable or allowing the use of backpropagation learning.

The choice of which activation function to use varies depending on the problem you want to solve, but let's look at some of them.

### ReLU

The ReLU activation function is widely used because it is computationally efficient,
is applied to a range of tasks, its advantage is precisely efficiency, as it is 0 for any negative number,
and returns the number itself for entries >= 0. A disadvantage is precisely the issue of returning 0, which can cause the problem
of the vanishing gradient, which impairs learning and can result in certain neurons not learning properly.

![Relu function](https://miro.medium.com/v2/resize:fit:1400/0*17a9Xr_jp1KXlxT8.png)

### Leaky ReLU

As a solution to the vanishing gradient problem, the Leaky ReLU function appears, which contains a leak so that the numbers
negatives are not taken as 0, but still making them very close to 0.

![Leaky ReLU](https://pytorch.org/docs/stable/_images/LeakyReLU.png)

### ELU

The idea of using ELU is to introduce non-linearity within our model, which can increase its complexity,
making it able to learn even more complex patterns, its difference to ReLU is that the negative values
are calculated from exponentiation.

![Elu](https://armandolivares.tech/wp-content/uploads/2022/09/elu-1.png)

### Linear

The linear function, as the name suggests, is a straight line, whose only function is to return the same value it receives,
being applicable to multiclass classification problems in the output and regression layers.

![Linear](https://cdn-academy.pressidium.com/academy/wp-content/uploads/2021/12/key-features-of-linear-function-graphs-2.png)


### Sigmoid

The sigmoid function is an activation function used for logistic regression, has a domain interval of (0, 1) and causes
very large values are close to 1 and very small values ​​are close to 0, normally it is indicated only in the output layers
of binary classification problems, as it does not have good computational performance in the intermediate layers, in addition to being very
prone to the vanishing gradient problem.

![Sigmoide](https://miro.medium.com/v2/resize:fit:1400/1*a04iKNbchayCAJ7-0QlesA.png)


### Tanh

The hyperbolic tangent activation function is an activation function that follows trigonometric behavior but also has
a curve similar to that of the sigmoid function, with the additional benefit of allowing negative values, in addition to having a derivative whose values ​​are not
as close to 0 as the sigmoid function.

![Tanh](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-05-27_at_4.23.22_PM_dcuMBJl.png)

### Softmax

Softmax is a function widely used in the context of multiclass classification, where it receives an input and converts it into a
probability distribution, where all probabilities sum to 1, is a generalization of logistic regression and is
widely applied whenever we need to predict 3 or more classes.

![Softmax](https://cdn.botpenguin.com/assets/website/Softmax_Function_07fe934386.png)
