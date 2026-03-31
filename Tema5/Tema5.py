import numpy as np


# ==========================================================
# JACOBI METHOD FOR EIGENVALUES / EIGENVECTORS
# ==========================================================

def max_offdiag(A):
    n = A.shape[0]
    max_val = 0
    p, q = 0, 1

    for i in range(n):
        for j in range(i):
            if abs(A[i, j]) > max_val:
                max_val = abs(A[i, j])
                p, q = i, j

    return p, q, max_val


def jacobi_method(A, eps=1e-10, kmax=1000):
    A = A.astype(float)
    n = A.shape[0]

    U = np.eye(n)
    A_init = A.copy()

    k = 0

    while k < kmax:
        p, q, max_val = max_offdiag(A)

        if max_val < eps:
            break

        if A[p, q] == 0:
            break

        alpha = (A[p, p] - A[q, q]) / (2 * A[p, q])

        t = np.sign(alpha) / (abs(alpha) + np.sqrt(alpha**2 + 1))
        if alpha == 0:
            t = 1

        c = 1 / np.sqrt(1 + t**2)
        s = t * c

        app = A[p, p]
        aqq = A[q, q]
        apq = A[p, q]

        # Update matrix A
        for j in range(n):
            if j != p and j != q:
                apj = A[p, j]
                aqj = A[q, j]

                A[p, j] = c * apj + s * aqj
                A[j, p] = A[p, j]

                A[q, j] = -s * apj + c * aqj
                A[j, q] = A[q, j]

        A[p, p] = app + t * apq
        A[q, q] = aqq - t * apq
        A[p, q] = 0
        A[q, p] = 0

        # Update U
        for i in range(n):
            uip = U[i, p]
            uiq = U[i, q]

            U[i, p] = c * uip + s * uiq
            U[i, q] = -s * uip + c * uiq

        k += 1

    eigenvalues = np.diag(A)

    Lambda = np.diag(eigenvalues)
    verification = np.linalg.norm(A_init @ U - U @ Lambda)

    return eigenvalues, U, verification


# ==========================================================
# CHOLESKY ITERATION
# ==========================================================

def cholesky_iteration(A, eps=1e-10, kmax=100):
    A = A.astype(float)

    for k in range(kmax):
        A_prev = A.copy()

        L = np.linalg.cholesky(A)
        A = L.T @ L

        diff = np.linalg.norm(A - A_prev)

        if diff < eps:
            break

    return A


# ==========================================================
# SVD CASE p > n
# ==========================================================

def svd_analysis(A):
    U, s, VT = np.linalg.svd(A, full_matrices=True)

    rank = np.linalg.matrix_rank(A)
    cond = np.linalg.cond(A)

    # Moore-Penrose pseudoinverse
    AI = np.linalg.pinv(A)

    # Least squares pseudoinverse
    AJ = np.linalg.inv(A.T @ A) @ A.T

    norm_diff = np.linalg.norm(AI - AJ, ord=1)

    return s, rank, cond, AI, AJ, norm_diff


# ==========================================================
# MAIN PROGRAM
# ==========================================================

def main():
    eps = 1e-10

    print("Introduce matricea A:")
    p = int(input("Numar linii p = "))
    n = int(input("Numar coloane n = "))

    A = []

    for i in range(p):
        row = list(map(float, input(f"Linia {i+1}: ").split()))
        A.append(row)

    A = np.array(A)

    print("\nMatricea A:")
    print(A)

    # ======================================================
    # CASE p = n and symmetric
    # ======================================================
    if p == n and np.allclose(A, A.T):

        print("\n=== METODA JACOBI ===")

        eigenvalues, eigenvectors, verification = jacobi_method(A, eps)

        print("\nValori proprii:")
        print(eigenvalues)

        print("\nVectori proprii:")
        print(eigenvectors)

        print("\nNorma ||A_init * U - U * Lambda||:")
        print(verification)

        print("\n=== ITERATIE CHOLESKY ===")

        try:
            A_final = cholesky_iteration(A, eps)

            print("\nUltima matrice calculata:")
            print(A_final)

        except np.linalg.LinAlgError:
            print("Matricea nu este pozitiv definita -> Cholesky imposibil.")

    # ======================================================
    # CASE p > n
    # ======================================================
    elif p > n:

        print("\n=== SVD ===")

        s, rank, cond, AI, AJ, norm_diff = svd_analysis(A)

        print("\nValori singulare:")
        print(s)

        print("\nRang:")
        print(rank)

        print("\nNumar de conditionare:")
        print(cond)

        print("\nPseudo-inversa Moore-Penrose AI:")
        print(AI)

        print("\nPseudo-inversa least squares AJ:")
        print(AJ)

        print("\nNorma ||AI - AJ||1:")
        print(norm_diff)

    else:
        print("\nCaz neacoperit explicit in tema.")


if __name__ == "__main__":
    main()