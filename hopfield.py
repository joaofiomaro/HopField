import glob
import os

import numpy as np
import openpyxl

# Limite de iterações: impede laço infinito caso a rede nunca estabilize.
MAX_EPOCAS = 100

# Dimensões da imagem original (5 linhas x 6 colunas = 30 componentes).
LINHAS, COLUNAS = 5, 6


def carregar_amostras(caminho):
    """Lê a planilha e separa os padrões armazenados do padrão desconhecido."""
    planilha = openpyxl.load_workbook(caminho, data_only=True).active

    # min_row=2 descarta a linha de cabeçalho da planilha.
    linhas = [list(r) for r in planilha.iter_rows(min_row=2, values_only=True)]
    dados = np.array(linhas, dtype=int)

    # Conforme o enunciado, a última linha é o padrão desconhecido.
    return dados[:-1], dados[-1]


def calcular_pesos(padroes):
    """Passo 1: monta a matriz de pesos W = soma dos produtos entre padrões."""
    # O produto padroes.T @ padroes já executa a somatória sobre os M padrões, substituindo os laços aninhados de i, j e r.
    W = padroes.T @ padroes

    # Regra w_ii = 0: nenhum neurônio se realimenta a si mesmo.
    np.fill_diagonal(W, 0)

    return W


def ativacao(u, anterior):
    """Não-linearidade tipo relé: +1 se u > 0, -1 se u < 0, mantém se u = 0."""
    return np.where(u > 0, 1, np.where(u < 0, -1, anterior))


def recuperar(W, padrao_inicial):
    """Passos 2 e 3: inicia a rede e itera até a saída deixar de mudar."""
    y = padrao_inicial.copy()
    epocas = 0

    while epocas < MAX_EPOCAS:
        # Todos os neurônios são atualizados a partir do estado anterior.
        novo = ativacao(W @ y, y)
        epocas += 1

        # Critério de parada: y(t+1) idêntico a y(t) (rede estabilizada).
        if np.array_equal(novo, y):
            return y, epocas, True

        y = novo

    return y, epocas, False


def identificar(y, padroes):
    """Compara o estado final com cada padrão e devolve o mais próximo."""
    # Divergências = quantidade de componentes diferentes em cada comparação.
    divergencias = [int(np.sum(y != p)) for p in padroes]
    indice = int(np.argmin(divergencias))

    return indice, divergencias[indice], divergencias


def desenhar(vetor):
    """Exibe o vetor de 30 posições como a imagem 5x6 que o originou."""
    for i in range(LINHAS):
        linha = vetor[i * COLUNAS:(i + 1) * COLUNAS]
        print("   ", "  ".join("#" if v == 1 else "." for v in linha))


# parte 1

print("TREINAMENTO DA REDE HOPFIELD\n")

arquivo = input("Arquivo Excel [Dados.xlsx]: ").strip() or "Dados.xlsx"

# Se o nome informado não existir, usa a primeira planilha da pasta atual.
if not os.path.isfile(arquivo):
    encontrados = glob.glob("*.xlsx")
    if not encontrados:
        raise SystemExit("Nenhum arquivo .xlsx encontrado nesta pasta.")
    arquivo = encontrados[0]
    print(f"Arquivo nao localizado; usando '{arquivo}'.")

padroes, desconhecido = carregar_amostras(arquivo)
print(f"\n{len(padroes)} padroes armazenados, {padroes.shape[1]} neuronios.")

W = calcular_pesos(padroes)

# linewidth alto evita que o numpy quebre a matriz em blocos separados.
np.set_printoptions(linewidth=250)
print(f"\nMatriz de pesos obtida ({W.shape[0]}x{W.shape[1]}):\n")
print(W)

# Conferência das duas propriedades exigidas pelo algoritmo.
print(f"\nSimetrica (w_ij = w_ji): {np.array_equal(W, W.T)}")
print(f"Diagonal nula (w_ii = 0): {not np.any(np.diag(W))}")


# parte 2

print("\nTESTE DO PADRAO DESCONHECIDO\n")
print("Padrao desconhecido lido da ultima linha da planilha:")
desenhar(desconhecido)

y, epocas, estabilizou = recuperar(W, desconhecido)

print("\nPadrao recuperado pela rede:")
desenhar(y)

indice, erro, todas = identificar(y, padroes)

print(f"\nEpocas necessarias: {epocas}")
print(f"Estabilizou: {'sim' if estabilizou else 'nao (limite de epocas)'}")
print(f"Divergencias por padrao armazenado: {todas}")

# Erro zero significa que o estado final é exatamente um padrão armazenado.
if erro == 0:
    print(f"\nAssociado ao PADRAO {indice + 1}")
    print("Erro cometido: 0 componentes (0.00%) - associacao exata")
else:
    print(f"\nNao corresponde exatamente a nenhum padrao armazenado.")
    print(f"Padrao mais proximo: PADRAO {indice + 1}")
    print(f"Erro cometido: {erro} de {len(y)} componentes "
          f"({100 * erro / len(y):.2f}%)")

print("\nPadrao mais proximo, para comparacao:")
desenhar(padroes[indice])