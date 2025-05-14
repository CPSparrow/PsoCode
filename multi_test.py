import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from pso import PsoModel, f,schedule


def mulit_test(n_worker: int, n_iter: int, n_test: int,schedule_type=None) -> np.ndarray:
    gb_list = []
    for i in tqdm(range(n_test)):
        model = PsoModel(n_worker, n_iter)
        _, gb = model.train(schedule_type)
        gb_list.append(f(gb[0], gb[1]))
    return np.array(gb_list)


if __name__ == "__main__":
    result_list=mulit_test(n_worker=50,n_iter=500,n_test=200,schedule_type="linear")
    print(f"avg:{result_list.mean():.3f}, std:{result_list.std():.3f}")
    print(f"min:{result_list.min():.3f}, max:{result_list.max():.3f}")

    # plt.bar(range(len(result_list)), result_list)
    # plt.ylim(bottom=result_list.min()*0.92)
    # plt.show()
