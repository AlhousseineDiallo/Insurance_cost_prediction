import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


class linear_regression:
    def __init__(self, x: NDArray[np.float64], seed=123):
        m, n = x.shape
        np.random.seed(seed=seed)
        self._w = np.random.rand(m + 1, 1)

    def predict(self, x: NDArray[np.float64]):
        """Prédictions du modèle.

        Args:
            x: Matrice de features (m, n)

        Returns:
            Prédictions (m, 1)
        """
        m, n = x.shape
        x_1 = np.hstack((np.ones((m, 1)), x))

        return np.dot(x_1, self._w)

    def compute_cost(self, y: NDArray[np.float64], y_hat: NDArray[np.float64]):
        """Coût MSE du modèle.

        Args:
            y: Vraies valeurs (m, 1)
            y_hat: Prédictions (m, 1)

        Returns:
            Coût (1, 1)
        """
        m, _ = y.shape
        cost = 1 / (2 * m) * np.dot((y - y_hat).T, (y - y_hat))
        return cost

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        learning_rate: float = 0.01,
        num_iters: int = 200,
    ) -> NDArray[np.float64]:
        m, n = x.shape
        x_1: NDArray[np.float64] = np.hstack((np.ones((m, 1)), x))
        # cette variable va nous permettre de garder les valeurs prises par notre fonction de coût
        # et ainsi avoir la learning curve de notre modèle
        J_history: NDArray[np.float64] = np.zeros((num_iters))

        # maintenant on va procéder au calcul de nos paramètres et donc a la mise a jour du modèle
        for i in range(num_iters):
            self._w = self._w - (learning_rate / m) * np.dot(
                x_1.T, (self.predict(x) - y)
            )

            J_history[i] = self.compute_cost(self.predict(x), y)

        return J_history
