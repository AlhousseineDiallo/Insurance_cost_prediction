import numpy as np
from numpy.typing import NDArray


class logistic_regression:
    def __init__(self, x: NDArray[np.float64], seed: int = 123):
        # initialisation des poids du modèles
        m, n = x.shape
        np.random.seed(seed=seed)
        self._w = np.random.rand(n + 1, 1)

    # definition de la fonction sigmoïde
    # rappelons que cette fonction aura pour rôle de borner les sorties linéaires du modèle entre 0 et 1

    def sigmoid(self, z: NDArray[np.float64]):
        # évitons une potentiel explosion de nos valeurs en restreignant le tout a l'intervalle -500 à 500
        z = np.clip(a=z, a_min=-500, a_max=500)
        return 1 / (1 + np.exp(-z))

    def predict(self, x: NDArray[np.float64]):
        # ici le but sera d'appliquer la fonction sigmoide a la sortie linéaire du modèle donc notre logit

        m, _ = x.shape
        # pour isoler correctement notre biais nous allons procéder a une petite manipulation
        x_1: NDArray[np.float64] = np.hstack(tup=(np.ones(shape=(m, 1)), x))

        # voici donc la sortie linéaire du modèle
        y_1 = np.dot(x_1, self._w)
        return self.sigmoid(z=y_1)

    def compute_cost(self, y: NDArray[np.float64], y_hat: NDArray[np.float64]):
        # le but ici sera de calculer la perte de notre modèle
        m = y.shape[0]

        # ajout d'un reel a y_hat pour eviter log(0) qui tendrait vers - l'infini
        epsillon = 1e-15
        y_hat = np.clip(a=y_hat, a_min=epsillon, a_max=1 - epsillon)

        # on va utiliser le produit matriciel pour automatiser la somme
        term_1 = np.dot(y.T, np.log(y_hat))
        term_2 = np.dot((1 - y).T, np.log(1 - y_hat))

        cost = -1 / m * (term_1 + term_2)

        return cost

    def fit(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        num_iters=200,
        learning_rate=0.001,
    ):
        m, _ = x.shape

        x_1 = np.hstack(tup=(np.ones(shape=(m, 1)), x))
        J_history = np.zeros(shape=num_iters)

        for i in range(0, num_iters):
            self._w = self._w - learning_rate / m * np.dot(x_1.T, (self.predict(x) - y))

            J_history[i] = self.compute_cost(y, self.predict(x))

        return J_history
