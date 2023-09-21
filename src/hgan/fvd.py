import numpy as np
import scipy
import tensorflow as tf


# FVD implementation mostly copied from
# https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py
# with minor tweaks to work with numpy instead of tensorflow


def _symmetric_matrix_square_root(mat, eps=1e-10):
    """Compute square root of a symmetric matrix.

    Note that this is different from an elementwise square root. We want to
    compute M' where M' = sqrt(mat) such that M' * M' = mat.

    Also note that this method **only** works for symmetric matrices.

    Args:
      mat: Matrix to take the square root of.
      eps: Small epsilon such that any element less than eps will not be square
        rooted to guard against numerical instability.

    Returns:
      Matrix square root of mat.
    """
    u, s, vT = scipy.linalg.svd(mat)
    # sqrt is unstable around 0, just use 0 in such case
    si = np.where(s < eps, s, np.sqrt(s))
    return np.matmul(np.matmul(u, tf.linalg.tensor_diag(si)), vT)


def trace_sqrt_product(sigma, sigma_v):
    """Find the trace of the positive sqrt of product of covariance matrices.

    '_symmetric_matrix_square_root' only works for symmetric matrices, so we
    cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
    ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).

    Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
    We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
    Note the following properties:
    (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
       => eigenvalues(A A B B) = eigenvalues (A B B A)
    (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
       => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
    (iii) forall M: trace(M) = sum(eigenvalues(M))
       => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                     = sum(sqrt(eigenvalues(A B B A)))
                                     = sum(eigenvalues(sqrt(A B B A)))
                                     = trace(sqrt(A B B A))
                                     = trace(sqrt(A sigma_v A))
    A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
    use the _symmetric_matrix_square_root function to find the roots of these
    matrices.

    Args:
      sigma: a square, symmetric, real, positive semi-definite covariance matrix
      sigma_v: same as sigma

    Returns:
      The trace of the positive square root of sigma*sigma_v
    """

    # Note sqrt_sigma is called "A" in the proof above
    sqrt_sigma = _symmetric_matrix_square_root(sigma)

    # This is sqrt(A sigma_v A) above
    sqrt_a_sigmav_a = np.matmul(sqrt_sigma, np.matmul(sigma_v, sqrt_sigma))

    return np.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))


def compute_fvd(real_activations: np.ndarray, generated_activations: np.ndarray):
    m = real_activations.mean(axis=0)
    m_w = generated_activations.mean(axis=0)
    n_samples = real_activations.shape[0]

    # sigma = (1 / (n - 1)) * (X - mu) (X - mu)^T
    real_centered = real_activations - m
    sigma = np.matmul(real_centered.T, real_centered) / (n_samples - 1)

    gen_centered = generated_activations - m_w
    sigma_w = np.matmul(gen_centered.T, gen_centered) / (n_samples - 1)

    # Find the Tr(sqrt(sigma sigma_w)) component of FID
    sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)

    # Compute the two components of FID.

    # First the covariance component.
    # Here, note that trace(A + B) = trace(A) + trace(B)
    trace = np.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

    # Next the distance between means
    mean = np.square(m - m_w).sum()

    fid = trace + mean
    assert fid >= 0, "Unexpected condition!"

    return float(fid)
