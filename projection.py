from math import ceil
from typing import Optional, overload

import numpy as np
import torch


class FeatureProjection:
    @property
    def P(self) -> torch.Tensor:
        return self.__P

    @P.setter
    def P(self, value: torch.Tensor) -> None:
        self.__P = value
        self.use_gpu = self.use_gpu

    @property
    def dim(self) -> int:
        return self.__dim

    @property
    def logging(self) -> bool:
        return self.__logging

    @logging.setter
    def logging(self, value: bool) -> None:
        self.__logging = value

    @property
    def use_gpu(self) -> bool:
        return self.__use_gpu

    @use_gpu.setter
    def use_gpu(self, value: bool) -> None:
        self.__use_gpu = value
        if self.__use_gpu:
            self.__P = self.__P.cuda()
        else:
            self.__P = self.__P.cpu()

    def __init__(self, dim: int, logging: bool = False, use_gpu: bool = False):
        self.__dim = dim
        self.__use_gpu = use_gpu
        self.__logging = logging
        self.__P = torch.empty(())

    def fit(self, X: np.ndarray, y: np.ndarray, n_epochs: int, batch_size: int):
        mean = np.mean(X, 0)
        std = np.std(X, 0)
        X = (X - mean) / (std + 1e-24)

        P = torch.from_numpy(np.empty((X.shape[1] + 1, self.dim), X.dtype))
        O = torch.empty((self.dim, y.max() + 1), dtype=P.dtype)

        torch.nn.init.kaiming_uniform_(P)
        torch.nn.init.xavier_uniform_(O)

        if self.use_gpu:
            P = P.cuda()
            O = O.cuda()

        P.requires_grad_()
        O.requires_grad_()

        opt = torch.optim.Adam([P, O])
        crit = torch.nn.CrossEntropyLoss(reduction="sum")

        batch_count = ceil(X.shape[0] / batch_size)
        for e in range(1, n_epochs + 1):
            running_loss = 0
            running_regularization_loss = 0
            self.__log(f"training epoch {e},  0%...", end="")
            for i in range(batch_count):
                start = i * batch_size
                end = start + batch_size
                feats = FeatureProjection.__homogenize(
                    torch.from_numpy(X[start:end, :]).to(P.device)
                )
                logits = feats @ P @ O
                target = torch.from_numpy(y[start:end]).to(P.device)
                loss = crit(logits, target)
                regularization_loss = (O * O).sum()
                running_loss += loss.item()
                running_regularization_loss += regularization_loss.item()
                (loss + regularization_loss).backward()
                opt.step()
                opt.zero_grad()
                self.__log(f"\rtraining epoch {e}, {i/batch_count:3.0%}", end="")
            self.__log(f"\r training epoch {e}, 100%", end="")
            running_loss /= X.shape[0]
            self.__log(f", loss: {running_loss:.8e}", end="")
            running_regularization_loss /= X.shape[0]
            self.__log(f", regularization loss: {running_regularization_loss:.8e}")

        P.requires_grad_(False)
        self.__P = torch.eye(X.shape[1] + 1, dtype=P.dtype)
        self.__P[-1, :-1] = -torch.from_numpy(mean)
        self.__P @= torch.from_numpy(
            np.diag(np.concat([1 / (std + 1e-24), np.ones((1,), std.dtype)]))
        )
        self.__P @= P

    @overload
    def __call__(self, X: np.ndarray) -> np.ndarray: ...

    @overload
    def __call__(self, X: torch.Tensor) -> torch.Tensor: ...

    @torch.no_grad
    def __call__(self, X: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(X, np.ndarray):
            return self(torch.from_numpy(X)).cpu().numpy()
        else:
            return FeatureProjection.__homogenize(X.to(self.__P.device)) @ self.__P

    def __log(self, message: str, end: Optional[str] = None) -> None:
        if self.logging:
            print(message, end=end)

    @staticmethod
    def __homogenize(X: torch.Tensor) -> torch.Tensor:
        return torch.concat(
            (X, torch.ones((X.shape[0], 1), dtype=X.dtype, device=X.device)), 1
        )
