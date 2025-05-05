import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

def f(x: np.ndarray, y: np.ndarray):
    return (
        6.452
        * (x + 0.125 * y)
        * np.pow((np.cos(x) - np.cos(2 * y)), 2)
        / np.sqrt(0.8 + np.pow(x - 4.2, 2) + 2 * np.pow(y - 7, 2))
        + 3.226 * y
    )


class PsoModel:
    def __init__(self, n_worker, n_iter):
        self.n_worker = n_worker
        self.n_iter = n_iter

        self.x_list = np.random.random((n_worker, 2)) * 10
        self.v_list = np.random.normal(loc=0, scale=1, size=(n_worker, 2))

        self.personal_best_list = self.x_list
        self.personal_score_list = f(self.x_list[:, 0], self.x_list[:, 1])

        best_index = np.argmax(self.personal_score_list)
        self.global_best = self.x_list[best_index]
        self.global_best_score = self.personal_score_list[best_index]

        self.gb_history = [self.global_best_score]

    def forward(self, omega=0.9, c1=2, c2=2):
        new_x = (self.x_list + self.v_list).clip(min=0, max=9.999999)

        new_score = f(new_x[:, 0], new_x[:, 1])
        self.personal_best_list = np.where(
            (new_score > self.personal_score_list)[:, None],
            new_x,
            self.personal_best_list,
        )
        self.personal_score_list = np.maximum(new_score, self.personal_score_list)

        best_index = np.argmax(self.personal_score_list)
        if new_score[best_index] > self.global_best_score:
            self.global_best = self.personal_best_list[best_index]
            self.global_best_score = self.personal_score_list[best_index]

        self.gb_history.append(self.global_best_score)

        self.v_list = (
            omega * self.v_list
            + c1
            * np.random.random((self.n_worker, 2))
            * (self.personal_best_list - self.x_list)
            + c2
            * np.random.random((self.n_worker, 2))
            * (self.global_best - self.x_list)
        )

    def train(self):
        for i in tqdm(range(self.n_iter), ncols=80, desc="Training:"):
            self.forward()
        return self.gb_history, self.global_best


if __name__ == "__main__":
    model = PsoModel(50, 500)
    history, gb = model.train()
    print(f"X:{gb[0]:.9f}, Y:{gb[1]:.9f}")
    print(f"score:{f(gb[0],gb[1]):.3f}")

    plt.figure(dpi=160)
    plt.plot(history)
    plt.title("PSO train log")
    plt.xlabel("iteration")
    plt.ylabel("score")
    plt.legend()
    plt.show()
