import numpy as np
import quadprog


def solve_QP(Q, c, A, b, E=None, d=None):
    """
        Solves the following quadratic program:
        minimize (1/2)x^T Q x + c^T x
        subject to Ax <= b and Ex=d

        for purpose of this assignment, E and d are NOT used.

        (Adapted from: https://scaron.info/blog/quadratic-programming-in-python.html)

        :param Q 2D Numpy matrix in the equation above
        :param c 1D Numpy matrix in the equation above
        :param A 2D Numpy matrix in the equation above
        :param b 1D Numpy matrix in the equation above
        :param E 2D Numpy matrix in the equation above
        :param d 1D  Numpy matrix in the equation above
        
        :return 1D Numpy array contaning the values of the variables in the optimal solution
    """

    # Perturb Q so it is positive definite
    qp_G = Q + 10 ** (-9) * np.identity(Q.shape[0])

    qp_a = -c
    if E is not None:
        qp_C = -np.vstack([E, A]).T
        qp_b = -np.hstack([d.T, b.T])
        meq = E.shape[0]
    else:  # no equality constraint
        qp_C = -A.T
        qp_b = -b
        meq = 0

    return quadprog.solve_qp(qp_G.astype(np.float64), qp_a.astype(np.float64), qp_C.astype(np.float64), qp_b.astype(np.float64), meq)[0]


def qp_example():
    """
    The only purpose of this example is to demonstrate how to use the QP solver.

    Solves the example available here: https://scaron.info/blog/quadratic-programming-in-python.html
    """
    Q = np.array([[6., 3.], [1., 2.]])
    c = np.array([-6., 4.]).reshape((2,))
    A = np.array([[-3., 1.], [0., 2.], [-1., 1.], [-1., 0.], [0., -1.]])
    b = np.array([-5., 6., -1., 0., 0.]).reshape((5,))
    print(solve_QP(Q, c, A, b))


# Run the example
if __name__ == "__main__":
    qp_example()
