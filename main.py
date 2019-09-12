import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets


def pca(X, k=30):
    """Centers X and applies principal component analysis (PCA) to reduce the dimensionality of X. This reduces the
    amount of memory and operations needed to calculate P. It also removes some noise from X (pg. 2589).

    Args:
        X: High-dimensional data set X, in the form [n=samples, N=number of high dimensions] (tensor)
        k: Number of PCA components, with k<<N (default: 30) (integer)
    """
    X = X/torch.max(X)  # Normalize X to between 0 and 1, prevents overflow when passed into torch.exp()
    X = X - torch.mean(X, dim=0)  # Center matrix for PCA
    U, S, Vt = torch.svd(X)
    X = torch.mm(U[:, 0:k], torch.diag(S[0:k]))  # Grab k principal components

    return X


def guassian_kernel_variance_search(variance, X, perp, vr, tol=0.01):
    """Estimate the Gaussian kernel variances using binary search to find a Shannon entropy value close to the given
    Perplexity parameter (pg. 2582). Note that variance is modified in-place as it has already been assigned to
    the graph.

    Args:
        variance: Vector of Gaussian kernel variances for each sample n (tensor)
        X: High-dimensional data set X, in the form [n=samples, N=number of high dimensions] (tensor)
        perp: Perplexity parameter
        vr: Gaussian kernel variance upper and lower limits, used for binary search
        tol: Tolerance for matching Shannon entropy with Perplexity
    """
    for i in range(n):
        variance_range = vr.clone()
        binary_search = True

        while binary_search:
            # Shannon entropy (bits)
            H = torch.exp(-torch.pow(torch.norm(X[i, :] - X, dim=1), 2) / (2*variance[i]))
            H[i] = 0
            H = H/torch.sum(H)

            # Value to match to Perplexity
            log_H = torch.log2(H)
            log_H[i] = 0
            H = torch.pow(2, -torch.sum(H*log_H))

            if H > perp + (tol*perp):
                variance_range[1] = variance[i]
                variance[i] = ((variance_range[1] - variance_range[0])/2) + variance_range[0]
            elif H < perp - (tol*perp):
                variance_range[0] = variance[i]
                variance[i] = ((variance_range[1] - variance_range[0])/2) + variance_range[0]
            else:
                binary_search = False


def estimate_p(X, variance):
    """Compute the pairwise affinities (conditional probabilites, p_j|i) using the Gaussian kernel function and
    estimated variances (Eq. 1). For P, each row i represents the conditional probability p_j|i. For p, probabilities
    are converted to a joint distribution, where p[i,j] is the probability of i and j.

    Args:
        X: High-dimensional data set X, in the form [n=samples, N=number of high dimensions] (tensor)
        variance: Vector of Gaussian kernel variances for each sample n (tensor)
    """
    # High-dimensional data set X conditional probabilities p_j|i in form P[i,j], P_i = P[i,:] (Eq. 1, pg. 2581)
    P = torch.exp(-torch.pow(torch.norm(X[:, None] - X[None, :], dim=2), 2)/(2*variance))  # Outer subtraction
    torch.diagonal(P)[:] = 0  # let p_i|i = 0
    P = P / torch.sum(P, dim=1).unsqueeze(1)  # Add fake dimension for broadcasting

    # Convert conditional probabilities to symmetrical conditional (joint) probabilities, p_ij = p[i,j] (pg. 2584)
    p = (P + torch.triu(P).T + torch.tril(P).T)/(2*n)
    torch.diagonal(p)[:] = float('nan')  # Set p_ii to nan (setting to 0 breaks auto gradients)

    return p


def estimate_q(Y, v):
    """Compute the low-dimensional affinities (joint probabilities, q[i,j]) using the t-distribution function (Eq. 4).
    For q, probabilities are converted to a joint distribution, where q[i,j] is the probability of i and j.

    Args:
        Y: Low-dimensional data set Y, in the form [n=samples, M=number of low dimensions] (tensor)
        v: Degrees of freedom for the t-distribution (integer)
    """
    # Low-dimensional data set Y joint probabilities q_ij in form q[i,j] (Eq. 4, pg. 2585)
    q = torch.pow(1 + torch.pow(torch.norm(Y[:, None] - Y[None, :], dim=2), 2), -(v+1)/2)  # Outer subtraction
    torch.diagonal(q)[:] = 0  # let q_i|i = 0
    q = q/torch.sum(q)
    torch.diagonal(q)[:] = float('nan')  # Set q_ii to nan (setting to 0 breaks auto gradients)

    return q


def tsne(X, perp=40, T=1000, lr=100, a=(0.5, 0.8), k=30, M=2, v=1, verbose=True):
    """t-Distributed Stochastic Neighbor Embedding of the high-dimensional data set X to the lower-dimensional data set
    Y of dimensionality M.

    Args:
        X: High-dimensional data set X, in the form [n=samples, N=number of high dimensions] (tensor)
        perp: Perplexity (Table 1, pg. 2589) (integer)
        T: Total number of gradient descent iterations (pg. 2588) (integer)
        lr: Learning rate (pg. 2588) (integer)
        a: Momentum  term for iterations 0:249 and 250:T (pg. 2588) (tuple)
        k: Number of principal components to reduce X (pg. 2589) (integer)
        M: Number of dimensions for low-dimensional data set Y (integer)
        v: Degrees of freedom for low-dimensional t-distribution (Eq. 4, pg. 2585) (integer)
        verbose: Output statistics about Gaussian kernel variance, gradient, and KL divergence loss value (bool)
    """

    # PCA to reduce dimensionality of X (pg. 2589)
    X = pca(X, k)

    # Generate initial map data points (y) for the low-dimensional data set Y (pg. 2582)
    Y = 10e-4 * torch.randn(n, M)  # Isotropic Gaussian

    # Generate initial Gaussian kernel variance estimates
    vr = torch.tensor([0, 50]).float()  # Magic number for upper and lower variance limits for binary search
    variance = (vr[1] - vr[0]) / 2 * torch.ones(n, 1)

    # Pass to GPU if possible
    if torch.cuda.is_available():
        X = X.cuda()
        Y = Y.cuda()
        variance = variance.cuda()
    Y.requires_grad = True  # Set last to avoid non-leaf tensors

    # Declare loss function and optimizer
    loss_function = nn.KLDivLoss(reduction='sum')
    optimizer = torch.optim.SGD([Y], lr=lr, momentum=a[0])

    # Find Gaussian kernel variance using binary search to match perplexity with a Shannon entropy value (pg. 2582)
    guassian_kernel_variance_search(variance, X, perp, vr, tol=0.01)
    if verbose:
        print('Gaussian kernel variance (mean):', torch.mean(variance).detach().cpu().numpy())
        print('Gaussian kernel variance (min):', torch.min(variance).detach().cpu().numpy())
        print('Gaussian kernel variance (max):', torch.max(variance).detach().cpu().numpy())

    # Estimate symmetrical conditional (joint) probabilities p[i,j] (pg. 2581 and 2584)
    p = estimate_p(X, variance)

    for t in range(T + 1):
        if t >= 250:  # Update momentum after 250 iterations
            optimizer = torch.optim.SGD([Y], lr=lr, momentum=a[1])

        optimizer.zero_grad()

        # Estimate joint probabilities q[i,j] (Eq. 4, pg. 2585)
        q = estimate_q(Y, v)

        # Early exaggeration regularization
        if t <= 50:
            loss = loss_function(torch.log(q), 4 * p)
        else:
            loss = loss_function(torch.log(q), p)

        # There is also a early compression regularizer, but no details were given on its selection of B.

        # Compute gradient and update map points in data set Y
        loss.backward()
        optimizer.step()

        if verbose and (t % 20) == 0:
            if t <= 50:
                print('Iteration:', t, '| KL Loss (exag.):', loss.detach().cpu().numpy(),
                      '| Gradient Norm (exag.):', torch.norm(Y.grad).detach().cpu().numpy())
            else:
                print('Iteration:', t, '| KL Loss:', loss.detach().cpu().numpy(),
                      '| Gradient Norm:', torch.norm(Y.grad).detach().cpu().numpy())

    return Y


"""
Parameters
"""
torch.manual_seed(1)  # Set random seed for reproducibility
perp = 40  # Perplexity (Table 1, pg. 2589)
T = 1000  # Total number of gradient descent iterations (pg. 2588)
lr = 100  # Learning rate (pg. 2588)
a = [0.5, 0.8]  # Momentum  term for iterations 0:249 and 250:T (pg. 2588)
k = 30  # Number of principal components to reduce X (pg. 2589)
n = 6000  # Number of random MNIST samples (pg. 2588)
M = 2  # Number of dimensions for low-dimensional data set Y
v = 1  # Degrees of freedom for low-dimensional t-distribution (Eq. 4, pg. 2585)
verbose = True  # Output statistics about Gaussian kernel variance, gradient, and KL divergence loss value

"""
Import high-dimensional data set X from MNIST test set
"""
mnist = datasets.MNIST('../data', train=False, download=True)
X = mnist.data.view(mnist.data.shape[0], -1).float()  # Flatten to vector and covert to single-precision
index = torch.randint(mnist.data.shape[0], (n, 1))  # Randomly select n images
X = X[index[:, 0], :]
targets = mnist.targets[index[:, 0]]

"""
Run t-SNE
"""
Y = tsne(X, perp, T, lr, a, k, M, v, verbose)

"""
Plot two-dimensional MNIST embedding
"""
fig, ax = plt.subplots()
for i0 in range(torch.max(targets).detach().cpu().numpy()):
    ax.plot(Y[targets == i0, 0].detach().cpu().numpy(), Y[targets == i0, 1].detach().cpu().numpy(), '.')
plt.show()
