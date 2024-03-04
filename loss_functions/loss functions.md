# Loss functions

## Português

Funções de perda são importantes para podermos não só informarmos a nossa rede sobre o quanto ela está errando, mas também
para quantificarmos esse erro e podermos atualizar os pesos corretamente, já que o backpropagation utiliza a derivada do erro
para realizar essa atualização de pesos.

Vamos conhecer algumas funções de perda e entender melhor sobre sua relação com as funções de ativação da camada de saída.

### Funções de perda para regressão

#### Erro Absoluto Médio

O Erro Absoluto Médio é uma medida de erro para problemas de regressão, não é tão sensível a outliers e não penaliza tão bem para erros maiores,
porém é mais fácil de interpretar uma vez que estará na mesma escala dos dados.

#### Erro Quadrático Médio

O Erro Quadrático Médio é uma medida de erro que é mais sensível aos outliers e que também pode penalizar bastante o modelo para erros muito grandes,
no entanto a interpretação de seus valores não é tão simples assim.


### Funções de perda para classificação

#### Entropia cruzada binária

Essa função de erro serve para calcularmos o erro em problemas de classificação binária, geralmente está associada à utilização de
uma função sigmoide na saída, essa função mede a diferença entre duas distribuições de probabilidade, sendo utilizada para validar
a distribuição gerada por nosso modelo e a distribuição dos dados que queremos prever.

#### Entropia cruzada categórica

É uma generalização da entropia cruzada categórica para problemas de classificação multiclasse.

## Quando utilizar?

Normalmente utilizamos EAM e EQM em problemas de regressão, atrelados a funções de ativação lineares na camada de saída, enquanto
que ECB e ECC são utilizadas em problemas de classificação, ECB em problemas com sigmoide na saída e ECC em problemas com uma linear
ou softmax na saída.
