import numpy as np

eps = 1e-12

def generate_spd_matrix(n):
    B = np.random.randn(n, n)
    A = B @ B.T #inmultie B cu Bt
    return A

def descompunere_ldlt(A):
    n = len(A)
    d = np.zeros(n) 

    for p in range(n):

        # calcul dp
        sum_val = 0.0
        for k in range(p):
            sum_val += d[k] * (A[p][k] ** 2)

        d[p] = A[p][p] - sum_val

        if abs(d[p]) < eps:
            raise ValueError("Matricea nu este pozitiv definita")

        # calcul l_ip
        for i in range(p+1, n):
            sum_val = 0.0
            for k in range(p):
                sum_val += d[k] * A[i][k] * A[p][k]

            A[i][p] = (A[i][p] - sum_val) / d[p]

    return d

#Lz=b L are 1 pe diag
def substitutie_directa(A, b):
    n = len(b)
    z = np.zeros(n)

    for i in range(n):
        sum_val = 0.0
        for j in range(i):
            sum_val += A[i][j] * z[j]
        z[i] = b[i] - sum_val

    return z

def diagonala(d, z):
    return z / d

def substitutie_inversa(A, y):
    n = len(y)
    x = np.zeros(n)

    for i in reversed(range(n)):
        sum_val = 0.0
        for j in range(i+1, n):
            sum_val += A[j][i] * x[j]
        x[i] = y[i] - sum_val

    return x

def determinantD(d):
    return np.prod(d)


def main():

    n = int(input("n = "))

    # generare matrice si vector
    A = generate_spd_matrix(n)
    A_init = A.copy() #pt verficare
    b = np.random.randn(n)

    
    x_lib = np.linalg.solve(A_init, b)

    d = descompunere_ldlt(A)

    detA = determinantD(d)
    # L si Lt au 1 pe diag principala deci det=1

    # rezolvare sistem
    z = substitutie_directa(A, b)
    y = diagonala(d, z)
    x_chol = substitutie_inversa(A, y)

    # norma
    residual = np.linalg.norm(A_init @ x_chol - b)
    error = np.linalg.norm(x_chol - x_lib)

    print("Determinant =", detA)
    print("||A xChol - b|| =", residual)
    print("||xChol - xlib|| =", error)


if __name__ == "__main__":
    main()