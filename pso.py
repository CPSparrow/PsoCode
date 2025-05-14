from typing import Literal
from time import time

import numpy as np


def f(x: np.ndarray, y: np.ndarray):
    return (
        6.452
        * (x + 0.125 * y)
        * np.pow((np.cos(x) - np.cos(2 * y)), 2)
        / np.sqrt(0.8 + np.pow(x - 4.2, 2) + 2 * np.pow(y - 7, 2))
        + 3.226 * y
    )


def schedule(
    current_iter: int, total_iter: int, schedule_type: Literal["cos", "linear", "mul"]
) -> float:
    omega_max, omega_min = 0.9, 1e-3
    if schedule_type == "linear":
        return omega_max - (omega_max - omega_min) * current_iter / total_iter
    elif schedule_type == "cos":
        return omega_min + 0.5 * (omega_max - omega_min) * (
            1 + np.cos(np.pi * current_iter / total_iter)
        )
    else:
        raise NotImplementedError(f"schedule_type {schedule_type} not implemented")


class PsoModel:
    def __init__(self, n_worker: int, n_iter, v_init=None):
        self.n_worker = n_worker
        self.n_iter = n_iter

        self.x_list = np.random.random((n_worker, 2)) * 10
        if v_init is None:
            self.v_list = np.random.normal(loc=0, scale=1, size=(n_worker, 2))
        elif isinstance(v_init, float):
            self.v_list = np.random.normal(loc=0, scale=v_init, size=(n_worker, 2))
        elif v_init == "uniform":
            self.v_list = np.random.uniform(low=-1, high=1, size=(n_worker, 2))
        elif v_init == "zero":
            self.v_list = np.zeros((n_worker, 2))
        else:
            raise NotImplementedError(f"v type {v_init} not implemented")

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

    def train(self, schedule_type=None):
        for i in range(self.n_iter):
            if schedule_type is not None:
                self.forward(schedule(i, self.n_iter, schedule_type))
            else:
                self.forward()
        return self.gb_history, self.global_best


if __name__ == "__main__":
    a = time()

    model = PsoModel(50, 500)
    history, gb = model.train(schedule_type="cos")
    print(f"X:{gb[0]:.9f}, Y:{gb[1]:.9f}")
    print(f"score:{f(gb[0],gb[1]):.3f}")

    print(f"time cost:{time()-a:.3f}")
