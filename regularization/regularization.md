# Regularization

## Português

Em determinados cenários, nosso modelo pode se sobreajustar aos dados ou não ter uma boa capacidade de generalização
dessa forma, precisamos aplicar algumas técnicas de regularização que façam com que nosos modelo aprenda a generalizar retirando
algumas informações dele.

### Dropout

Na técnica de dropout nós utilizamos uma probabilidade de que alguns neurônios na nossa rede sejam ignorados no processo de aprendizado,
dessa forma nossa rede aprenda a realizar as suas predições e se ajusta aos dados sem utilizar todos os seus neurônios,
aumentando sua capacidade de generalização.

### Regularização L1

Na regularização nós adicionamos uma penalização à função de custo, de forma que a rede se torna esparsa, ou seja algumas sinapses serão zeradas,
de forma que a rede irá descartar os neurônios que foram zerados, ou seja, "matando" neurônios da nossa rede que não contribuam devidamente para
o aprendizado.

### Regularização L2

Neste tipo de regularização nós modificamos o termo da regularização L1, de forma que o que agora é utilizado se torna um
decaimento de pesos, ou seja, a rede é penalizada a partir do momento em que pesos muito grandes são utilizados para realizar
as predições e toda a modelagem dos dados.

## English

In certain scenarios, our model may overfit the data or not have good generalization ability
Therefore, we need to apply some regularization techniques that make our model learn to generalize by removing
some information about him.

### Dropout

In the dropout technique we use a probability that some neurons in our network will be ignored in the learning process,
This way, our network learns to make its predictions and adjusts to the data without using all of its neurons,
increasing its generalization capacity.

### L1 regularization

In regularization we add a penalty to the cost function, so that the network becomes sparse, that is, some synapses will be zeroed,
so that the network will discard the neurons that were reset, that is, "killing" neurons in our network that do not contribute properly to
the learning.

### L2 regularization

In this type of regularization we modify the L1 regularization term, so that what is now used becomes a
weight decay, that is, the network is penalized from the moment that very large weights are used to carry out
predictions and all data modeling.
