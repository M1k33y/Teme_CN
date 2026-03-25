import numpy as np

# ex1 - inmultire matrice vector
def inmultire_matrice_vector(A, x):
    n = len(A)
    b = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i][j] * x[j]
        b[i] = s
    return b


# substitutie inversa pentru sistem triunghiular superior
def substitutie_inversa(R, y):
    n = len(y)
    x = np.zeros(n)
    for i in reversed(range(n)):
        s = 0.0
        for j in range(i + 1, n):
            s += R[i][j] * x[j]
        x[i] = (y[i] - s) / R[i][i]
    return x


n = int(input("n = "))
eps = 1e-12

A = np.random.randint(1, 10, (n, n)).astype(float)
s = np.random.randint(1, 10, n).astype(float)

# ex1 - calcul b = A * s
b = inmultire_matrice_vector(A, s)

print("Matricea A:")
print(A)
print("\nVectorul s:")
print(s)
print("\nVectorul b:")
print(b)

A_init = A.copy()
b_init = b.copy()

U_list = []
beta_list = []

# ex2 - descompunere QR Householder
for r in range(n - 1):
    sigma = 0.0
    for i in range(r, n):
        sigma += A[i][r] ** 2

    if sigma <= eps:
        U_list.append(np.zeros(n))
        beta_list.append(0.0)
        continue

    if A[r][r] >= 0:
        k = -np.sqrt(sigma)
    else:
        k = np.sqrt(sigma)

    beta = sigma - k * A[r][r]

    u = np.zeros(n)
    u[r] = A[r][r] - k
    for i in range(r + 1, n):
        u[i] = A[i][r]

    # transformare A
    for j in range(r, n):
        s_val = 0.0
        for i in range(r, n):
            s_val += u[i] * A[i][j]
        s_val /= beta

        for i in range(r, n):
            A[i][j] -= s_val * u[i]

    # transformare b (QT * b)
    s_val = 0.0
    for i in range(r, n):
        s_val += u[i] * b[i]
    s_val /= beta

    for i in range(r, n):
        b[i] -= s_val * u[i]

    U_list.append(u)
    beta_list.append(beta)


# R si y
R = A
y = b.copy()

# ex3 - rezolvare sistem Rx = QT*b
x_householder = substitutie_inversa(R, y)

# solutie cu biblioteca
Q_lib, R_lib = np.linalg.qr(A_init)
x_qr = np.linalg.solve(R_lib, Q_lib.T @ b_init)

print("\n||x_qr - x_householder|| =",
      np.linalg.norm(x_qr - x_householder),
      "-> diferenta intre solutia din QR biblioteca si Householder")

# ex4 - erori
print("\n||A_init*x_householder - b_init|| =",
      np.linalg.norm(A_init @ x_householder - b_init),
      "-> err pentru solutia Householder")

print("||A_init*x_qr - b_init|| =",
      np.linalg.norm(A_init @ x_qr - b_init),
      "-> err pentru solutia QR din biblioteca")

print("||x_householder - s|| =",
      np.linalg.norm(x_householder - s),
      "-> err fata de solutia exacta (Householder)")

print("||x_qr - s|| =",
      np.linalg.norm(x_qr - s),
      "-> err fata de solutia exacta (QR biblioteca)")

# ex5 - calcul inversa
A_inv_house = np.zeros((n, n))

for j in range(n):
    e = np.zeros(n)
    e[j] = 1.0

    b_temp = e.copy()

    # aplicare transformari Householder pe e (QT * e)
    for r in range(n - 1):
        u = U_list[r]
        beta = beta_list[r]

        if beta == 0:
            continue

        s_val = 0.0
        for i in range(r, n):
            s_val += u[i] * b_temp[i]
        s_val /= beta

        for i in range(r, n):
            b_temp[i] -= s_val * u[i]

    x = substitutie_inversa(R, b_temp)
    A_inv_house[:, j] = x


A_inv_lib = np.linalg.inv(A_init)

print("\n||A_inv_householder - A_inv_lib|| =",
      np.linalg.norm(A_inv_house - A_inv_lib),
      "-> diferenta dintre inversa calculata cu Householder si cea din biblioteca")