import numpy as np

class Sigmoid:

    def forward(self, z: np.ndarray):
        """
        Returns elementwise sigmoid values.
        """
        self.out = np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))
        return self.out
        

    def backwards(self, dL: np.ndarray) -> np.ndarray:
        return dL * self.out * (1 - self.out)
        


class Linear:

    def __init__(self, dim):
        self.w = np.zeros(dim)
        self.wgrad = 0
        self.b = np.zeros((1, ))
        self.bgrad = 0

    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.w + self.b


    def backward(self, dS: np.ndarray, x: np.ndarray) -> np.ndarray:
        self.wgrad = x.T @ dS
        self.bgrad = dS.sum()


class Model:

    def __init__(self, dim):
        self.linear = Linear(dim)
        self.sigmoid = Sigmoid()

    
    def forward(self, x: np.ndarray) -> np.ndarray:
        lin_out = self.linear.forward(x)
        self.out = self.sigmoid.forward(lin_out)

        return self.out


    def backward(self, x: np.ndarray, target: np.ndarray):
        loss = _calculate_loss(self.out, target)
        n = target.size
        dL = (
            -target / self.out
            + (1 - target) / (1 - self.out)
        ) / n
        dS = self.sigmoid.backwards(dL)
        self.linear.backward(dS, x)


    def optimize(self, lr: float) -> None:
        self.linear.w -= lr * self.linear.wgrad
        self.linear.b -= lr * self.linear.bgrad

        
        
def _calculate_loss(out: np.ndarray, target: np.ndarray) -> np.ndarray:
    return -(1/len(out)) * (target * np.log(out) + (1 - target) * np.log(1 - out)).sum(axis=-1)
    #out should be (x_size, ), target is (x_size, ), output should be (1, )
 
    

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    input -> linear y = ax + b -> activation
    backwards 
    loss dL/dL = 1. dL/dsigma = 
    """
    # Write code here
    model = Model(X.shape[1])

    for _ in range(steps):
        out = model.forward(X)
        model.backward(X, y)
        model.optimize(lr)

    return model.linear.w, model.linear.b

        
    