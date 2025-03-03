from typing import Literal, cast, overload

import numpy as np
import torch


class KNNClassifier:
    @property
    def dist_fn(self) -> str:
        return self.__dist_fn

    @property
    def k(self) -> int:
        return self.__k

    @k.setter
    def k(self, value: int) -> None:
        self.__k = value

    def __init__(
        self,
        k: int,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        dist_fn: Literal["euclidean"] | Literal["cosine"] = "euclidean",
        use_gpu: bool = False,
    ):
        self.__k = k
        self.__X_train = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
        self.__y_train = torch.from_numpy(y) if isinstance(y, np.ndarray) else y
        self.__class_count = cast(int, self.__y_train.max().item()) + 1
        self.__dist_fn = dist_fn
        if use_gpu:
            self.__X_train = self.__X_train.cuda()
            self.__y_train = self.__y_train.cuda()
        if dist_fn == "cosine":
            self.__X_train = self.__X_train / self.__X_train.norm(dim=1, keepdim=True)

    @overload
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @overload
    def predict(self, X: torch.Tensor) -> torch.Tensor: ...

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        dists = self.compute_dists(X)
        k_best = self.compute_k_best_classes(dists)
        result_torch = self.decide(k_best)
        if isinstance(X, np.ndarray):
            return result_torch.cpu().numpy()
        else:
            return result_torch

    @torch.no_grad
    def compute_dists(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        X_torch = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
        X_torch = X_torch.to(self.__X_train.device)
        if self.__dist_fn == "euclidian":
            return KNNClassifier.__euclidian_distance(self.__X_train, X_torch)
        else:
            return KNNClassifier.__cosine_distance(self.__X_train, X_torch)

    @torch.no_grad
    def compute_k_best_classes(self, dists: torch.Tensor) -> torch.Tensor:
        return self.__y_train[(dists.argsort(1)[:, : self.k])]

    @torch.no_grad
    def decide(self, k_best: torch.Tensor) -> torch.Tensor:
        vote_counts = torch.empty(
            (k_best.shape[0], self.__class_count),
            dtype=torch.int32,
            device=k_best.device,
        )
        for i in range(self.__class_count):
            vote_counts[:, i] = (k_best == i).count_nonzero(dim=1)
        return vote_counts.argmax(1)

    @staticmethod
    def __euclidian_distance(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return (A.expand(B.shape[0], -1, -1) - B.reshape(B.shape[0], 1, -1)).norm(dim=2)

    @staticmethod
    def __cosine_distance(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return 1 - (B / B.norm(dim=1, keepdim=True)) @ A.T
