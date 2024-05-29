import numpy as np

def load_fm_model(model_txt, N, Nf):
    model = model_txt
    idx = 1
    idf = 1
    offset = []
    L = []
    V = [[] for _ in range(N)]
    with open(model, 'r') as f:
        for line in f:
            split_line = line.strip().split(' ')
            numbers = list(map(float, split_line[1:]))
            if idx == 1:
                offset.append(numbers)
            elif 1 < idx <= 1 + N:
                L.append(numbers)
            elif idx > 1 + N:
                V[idf - 1].append(numbers)
                if len(V[idf - 1]) == Nf:
                    idf += 1
            idx += 1

    offset = np.array(offset)
    L = np.array(L)
    V = np.array(V)

    Q = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                Q[i, j] = L[i]
            elif i < j:
                Q[i, j] = np.dot(V[i, j % Nf], V[j, i % Nf])
            else:
                Q[i, j] = 0
    return Q, offset
