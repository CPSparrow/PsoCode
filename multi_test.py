import numpy as np
from tqdm import tqdm

from pso import PsoModel, f


def mulit_test(
    n_worker: int, n_iter: int, n_test: int, schedule_type=None, init_type=None
) -> np.ndarray:
    gb_list = []
    for i in tqdm(range(n_test), ncols=80):
        model = PsoModel(n_worker, n_iter, v_init=init_type)
        _, gb = model.train(schedule_type)
        gb_list.append(f(gb[0], gb[1]))
    return np.array(gb_list)


if __name__ == "__main__":
    n_test = 200
    np.random.seed(42)
    result_list = mulit_test(
        n_worker=50, n_iter=500, n_test=n_test, schedule_type="cos", init_type=None
    )

    print(f"avg:{result_list.mean():.5f}, std: {result_list.std():.5f}")
    print(f"min:{result_list.min():.5f}, max:{result_list.max():.5f}")
    cnt = (result_list <= 80).astype(int).sum() / n_test
    print(f"less than 80:{cnt*100:.2f}")
    print(f"{result_list.max():.8f}")
