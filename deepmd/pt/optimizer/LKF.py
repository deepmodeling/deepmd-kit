# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import math

import torch
from torch.optim.optimizer import (
    Optimizer,
)

log = logging.getLogger(__name__)


class LKFOptimizer(Optimizer):
    def __init__(
        self,
        params,
        kalman_lambda=0.98,
        kalman_nue=0.9987,
        block_size=5120,
    ):
        defaults = {
            "lr": 0.1,
            "kalman_nue": kalman_nue,
            "block_size": block_size,
        }
        super().__init__(params, defaults)

        self._params = self.param_groups[0]["params"]

        if len(self.param_groups) != 1 or len(self._params) == 0:
            raise ValueError(
                "LKF doesn't support per-parameter options " "(parameter groups)"
            )

        # NOTE: LKF has only global state, but we register it as state for
        # the first param, because this helps with casting in load_state_dict
        self._state = self.state[self._params[0]]
        self._state.setdefault("kalman_lambda", kalman_lambda)

        self.__init_P()

    def __init_P(self):
        param_nums = []
        param_sum = 0
        block_size = self.__get_blocksize()
        data_type = self._params[0].dtype
        device = self._params[0].device

        for param_group in self.param_groups:
            params = param_group["params"]
            for param in params:
                param_num = param.data.nelement()
                if param_sum + param_num > block_size:
                    if param_sum > 0:
                        param_nums.append(param_sum)
                    param_sum = param_num
                else:
                    param_sum += param_num

        param_nums.append(param_sum)

        P = []
        params_packed_index = []
        log.info("LKF parameter nums: %s" % param_nums)
        for param_num in param_nums:
            if param_num >= block_size:
                block_num = math.ceil(param_num / block_size)
                for i in range(block_num):
                    if i != block_num - 1:
                        P.append(
                            torch.eye(
                                block_size,
                                dtype=data_type,
                                device=device,
                            )
                        )
                        params_packed_index.append(block_size)
                    else:
                        P.append(
                            torch.eye(
                                param_num - block_size * i,
                                dtype=data_type,
                                device=device,
                            )
                        )
                        params_packed_index.append(param_num - block_size * i)
            else:
                P.append(torch.eye(param_num, dtype=data_type, device=device))
                params_packed_index.append(param_num)

        self._state.setdefault("P", P)
        self._state.setdefault("weights_num", len(P))
        self._state.setdefault("params_packed_index", params_packed_index)

    def __get_blocksize(self):
        return self.param_groups[0]["block_size"]

    def __get_nue(self):
        return self.param_groups[0]["kalman_nue"]

    def __split_weights(self, weight):
        block_size = self.__get_blocksize()
        param_num = weight.nelement()
        res = []
        if param_num < block_size:
            res.append(weight)
        else:
            block_num = math.ceil(param_num / block_size)
            for i in range(block_num):
                if i != block_num - 1:
                    res.append(weight[i * block_size : (i + 1) * block_size])
                else:
                    res.append(weight[i * block_size :])
        return res

    def __update(self, H, error, weights):
        P = self._state.get("P")
        kalman_lambda = self._state.get("kalman_lambda")
        weights_num = self._state.get("weights_num")
        params_packed_index = self._state.get("params_packed_index")

        block_size = self.__get_blocksize()
        kalman_nue = self.__get_nue()

        tmp = 0
        for i in range(weights_num):
            tmp = tmp + (kalman_lambda + torch.matmul(torch.matmul(H[i].T, P[i]), H[i]))

        A = 1 / tmp

        for i in range(weights_num):
            K = torch.matmul(P[i], H[i])

            weights[i] = weights[i] + A * error * K

            P[i] = (1 / kalman_lambda) * (P[i] - A * torch.matmul(K, K.T))

        kalman_lambda = kalman_nue * kalman_lambda + 1 - kalman_nue
        self._state.update({"kalman_lambda": kalman_lambda})

        i = 0
        param_sum = 0
        for param_group in self.param_groups:
            params = param_group["params"]
            for param in params:
                param_num = param.nelement()
                weight_tmp = weights[i][param_sum : param_sum + param_num]
                if param_num < block_size:
                    if param.ndim > 1:
                        param.data = weight_tmp.reshape(
                            param.data.T.shape
                        ).T.contiguous()
                    else:
                        param.data = weight_tmp.reshape(param.data.shape)

                    param_sum += param_num

                    if param_sum == params_packed_index[i]:
                        i += 1
                        param_sum = 0
                else:
                    block_num = math.ceil(param_num / block_size)
                    for j in range(block_num):
                        if j == 0:
                            tmp_weight = weights[i]
                        else:
                            tmp_weight = torch.concat([tmp_weight, weights[i]], dim=0)
                        i += 1
                    param.data = tmp_weight.reshape(param.data.T.shape).T.contiguous()

    def set_grad_prefactor(self, grad_prefactor):
        self.grad_prefactor = grad_prefactor

    def step(self, error):
        params_packed_index = self._state.get("params_packed_index")

        weights = []
        H = []
        param_index = 0
        param_sum = 0

        for param in self._params:
            if param.ndim > 1:
                tmp = param.data.T.contiguous().reshape(param.data.nelement(), 1)
                if param.grad is None:
                    tmp_grad = torch.zeros_like(tmp)
                else:
                    tmp_grad = (
                        (param.grad / self.grad_prefactor)
                        .T.contiguous()
                        .reshape(param.grad.nelement(), 1)
                    )
            else:
                tmp = param.data.reshape(param.data.nelement(), 1)
                if param.grad is None:
                    tmp_grad = torch.zeros_like(tmp)
                else:
                    tmp_grad = (param.grad / self.grad_prefactor).reshape(
                        param.grad.nelement(), 1
                    )

            tmp = self.__split_weights(tmp)
            tmp_grad = self.__split_weights(tmp_grad)

            for split_grad, split_weight in zip(tmp_grad, tmp):
                nelement = split_grad.nelement()

                if param_sum == 0:
                    res_grad = split_grad
                    res = split_weight
                else:
                    res_grad = torch.concat((res_grad, split_grad), dim=0)
                    res = torch.concat((res, split_weight), dim=0)

                param_sum += nelement

                if param_sum == params_packed_index[param_index]:
                    H.append(res_grad)
                    weights.append(res)
                    param_sum = 0
                    param_index += 1

        self.__update(H, error, weights)
