import numpy as np


# ==========================================================
# CITIRE FISIERE
# ==========================================================

def read_vector(filename):
    with open(filename, 'r') as f:
        return np.array([float(line.strip()) for line in f])


def read_diagonal(filename):
    with open(filename, 'r') as f:
        lines = [float(line.strip()) for line in f]

    offset = int(lines[0])
    values = np.array(lines[1:])

    return offset, values


# ==========================================================
# VERIFICARE DIAGONALA PRINCIPALA
# ==========================================================

def verify_d0(d0, eps):
    return np.all(np.abs(d0) > eps)


# ==========================================================
# GAUSS-SEIDEL PE MATRICE RARA
# ==========================================================

def gauss_seidel_sparse(d0, p, d1, q, d2, b, eps=1e-8, kmax=10000):
    n = len(d0)

    x = np.zeros(n)

    for k in range(kmax):
        x_old = x.copy()

        for i in range(n):
            s = 0.0

            # diagonala p inferioara
            if i - p >= 0:
                s += d1[i - p] * x[i - p]

            # diagonala p superioara
            if i + p < n:
                s += d1[i] * x_old[i + p]

            # diagonala q inferioara
            if i - q >= 0:
                s += d2[i - q] * x[i - q]

            # diagonala q superioara
            if i + q < n:
                s += d2[i] * x_old[i + q]

            x[i] = (b[i] - s) / d0[i]

        delta = np.linalg.norm(x - x_old, ord=np.inf)

        if delta < eps:
            return x, True, k + 1

        if delta > 1e10:
            return x, False, k + 1

    return x, False, kmax


# ==========================================================
# CALCUL y = Ax
# ==========================================================

def compute_Ax(d0, p, d1, q, d2, x):
    n = len(d0)
    y = np.zeros(n)

    for i in range(n):
        y[i] += d0[i] * x[i]

        if i + p < n:
            y[i] += d1[i] * x[i + p]

        if i - p >= 0:
            y[i] += d1[i - p] * x[i - p]

        if i + q < n:
            y[i] += d2[i] * x[i + q]

        if i - q >= 0:
            y[i] += d2[i - q] * x[i - q]

    return y


# ==========================================================
# MAIN
# ==========================================================

def main():
    eps = 1e-8

    d0 = read_vector("d0.txt")
    p, d1 = read_diagonal("d1.txt")
    q, d2 = read_diagonal("d2.txt")
    b = read_vector("b.txt")

    n = len(d0)

    print("\nDimensiunea sistemului n = ", n)
    print("p =", p)
    print("q =", q)

    if not verify_d0(d0, eps):
        print("Exista element nul pe diagonala principala.")
        return

    print("\nToate elementele diagonalei principale sunt nenule.")

    x, converged, iterations = gauss_seidel_sparse(d0, p, d1, q, d2, b, eps)

    if converged:
        print("\nSolutie aproximativa Gauss-Seidel:")
        print(x)

        print("\nNumar iteratii:", iterations)

        y = compute_Ax(d0, p, d1, q, d2, x)

        print("\ny = Ax:")
        print(y)

        norm = np.linalg.norm(y - b, ord=np.inf)

        print("\nNorma ||AxGS - b||inf:")
        print(norm)

    else:
        print("\nMetoda divergenta.")


if __name__ == "__main__":
    main()
