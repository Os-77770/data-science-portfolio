import numpy as np


def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return (Z > 0).astype(float)


def softmax_from_logits(Z):
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z_shifted)
    return expZ / np.sum(expZ, axis=1, keepdims=True)


def log_softmax_from_logits(Z):
    
      
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(Z_shifted), axis=1, keepdims=True))
    return Z_shifted - logsumexp


def initialize_weights(layers, seed=42):
    np.random.seed(seed)
    weights = []

    for i in range(len(layers) - 1):
        W = np.random.uniform(-1, 1, (layers[i], layers[i + 1]))
        b = np.random.uniform(-1, 1, (1, layers[i + 1]))
        weights.append((W, b))

    return weights



def forward_prop(X, weights):
    A = X
    cache = []

    
    for i in range(len(weights) - 1):
        W, b = weights[i]
        A_prev = A
        Z = np.dot(A_prev, W) + b
        A = relu(Z)
        cache.append((A_prev, Z, A))

    
    W, b = weights[-1]
    A_prev = A
    Z = np.dot(A_prev, W) + b
    A = softmax_from_logits(Z)
    cache.append((A_prev, Z, A))

    return A, cache


def compute_loss(y_true, weights, cache, lambda_):
    m = y_true.shape[0]

    output_logits = cache[-1][1]
    log_probs = log_softmax_from_logits(output_logits)

    cross_entropy = -np.mean(log_probs[np.arange(m), y_true])

    l2 = sum(np.sum(W ** 2) for W, _ in weights)

    return cross_entropy + (lambda_ / (2 * m)) * l2



def back_prop(X, y, weights, cache, lambda_):
    m = X.shape[0]
    grads = [None] * len(weights)

    y_pred = cache[-1][2]

    y_onehot = np.zeros_like(y_pred)
    y_onehot[np.arange(m), y] = 1

    
    dZ = (y_pred - y_onehot) / m

    for i in reversed(range(len(weights))):
        A_prev, Z, A = cache[i]
        W, b = weights[i]

        dW = np.dot(A_prev.T, dZ) + (lambda_ / m) * W
        db = np.sum(dZ, axis=0, keepdims=True)
        grads[i] = (dW, db)

        if i > 0:
            dA_prev = np.dot(dZ, W.T)
            prev_Z = cache[i - 1][1]
            dZ = dA_prev * relu_derivative(prev_Z)

    return grads



def predict(X, weights):
    y_prob, _ = forward_prop(X, weights)
    return np.argmax(y_prob, axis=1)



def misclassification_error(y_true, y_pred):
    return np.mean(y_true != y_pred)



def gradient_descent(X, y, weights, lr, lambda_, iterations, verbose=True):
    best_weights = [(W.copy(), b.copy()) for W, b in weights]
    best_error = float("inf")

    for i in range(iterations):
        y_prob, cache = forward_prop(X, weights)
        grads = back_prop(X, y, weights, cache, lambda_)

        new_weights = []
        for (W, b), (dW, db) in zip(weights, grads):
            W_new = W - lr * dW
            b_new = b - lr * db
            new_weights.append((W_new, b_new))
        weights = new_weights

        preds = predict(X, weights)
        error = misclassification_error(y, preds)

        if error < best_error:
            best_error = error
            best_weights = [(W.copy(), b.copy()) for W, b in weights]

        if verbose and i % 50 == 0:
            print(f"Iteration {i}, Error: {error:.4f}")

    return best_weights



def train_and_record(X_train, y_train, X_test, y_test, layers, lr, lambda_, iterations):
    weights = initialize_weights(layers)

    train_losses = []
    test_losses = []
    train_errors = []
    test_errors = []

    best_weights = [(W.copy(), b.copy()) for W, b in weights]
    best_train_error = float("inf")

    for i in range(iterations):
        y_prob_train, cache_train = forward_prop(X_train, weights)
        y_prob_test, cache_test = forward_prop(X_test, weights)

        train_loss = compute_loss(y_train, weights, cache_train, lambda_)
        test_loss = compute_loss(y_test, weights, cache_test, lambda_)

        train_pred = np.argmax(y_prob_train, axis=1)
        test_pred = np.argmax(y_prob_test, axis=1)

        train_err = misclassification_error(y_train, train_pred)
        test_err = misclassification_error(y_test, test_pred)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_errors.append(train_err)
        test_errors.append(test_err)

        if train_err < best_train_error:
            best_train_error = train_err
            best_weights = [(W.copy(), b.copy()) for W, b in weights]

        grads = back_prop(X_train, y_train, weights, cache_train, lambda_)

        new_weights = []
        for (W, b), (dW, db) in zip(weights, grads):
            W_new = W - lr * dW
            b_new = b - lr * db
            new_weights.append((W_new, b_new))
        weights = new_weights

        if i % 50 == 0:
            print(f"Iteration {i}, Train Error: {train_err:.4f}, Test Error: {test_err:.4f}")

    return best_weights, train_losses, test_losses, train_errors, test_errors