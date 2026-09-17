# Rede de Hopfield

Rede neural de Hopfield que armazena padrões binários (imagens 5x6, ou seja,
vetores de 30 componentes com valores +1 e -1) e recupera o padrão correto a
partir de uma entrada desconhecida.

## Arquivos

- `hopfield.py` — código-fonte (treinamento e teste)
- `Dados.xlsx` — planilha com 1 linha de cabeçalho, 7 padrões armazenados e o
  padrão desconhecido na última linha

## Requisitos

- Python 3.8 ou superior
- numpy e openpyxl

Para instalar as bibliotecas:

```
pip install numpy openpyxl
```

## Como executar

Na pasta do projeto:

```
python hopfield.py
```

O programa pede o nome da planilha. Basta pressionar Enter para usar o
`Dados.xlsx`.

## O que o programa faz

Parte 1 — Treinamento:

- Lê a planilha e separa os padrões armazenados do padrão desconhecido
- Calcula a matriz de pesos W, com a diagonal zerada (w_ii = 0)
- Mostra a matriz 30x30 e confere se ela é simétrica e tem diagonal nula

Parte 2 — Teste:

- Mostra o padrão desconhecido desenhado como imagem 5x6 (`#` para +1 e `.`
  para -1)
- Itera y(t+1) = f(W · y(t)) até a saída parar de mudar (limite de 100 épocas)
- Compara o resultado com cada padrão armazenado e informa o número de épocas,
  as divergências, o padrão associado e o erro cometido

## Saída obtida com o Dados.xlsx deste repositório

- Épocas necessárias: 5
- Estabilizou: sim
- Divergências por padrão armazenado: [9, 21, 10, 11, 15, 15, 1]
- Padrão mais próximo: PADRÃO 7
- Erro cometido: 1 de 30 componentes (3,33%)

## Usando outra planilha

Manter o mesmo formato: primeira linha de cabeçalho, uma linha por padrão com
valores +1 e -1, e o padrão a ser reconhecido na última linha.
