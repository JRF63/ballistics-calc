from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class NumericalIntegrator(ABC):
    @abstractmethod
    def __init__(
        self,
        x: np.ndarray,
        v: np.ndarray,
        accel_func: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        pass

    @abstractmethod
    def step(
        self,
        accel_func: Callable[[np.ndarray], np.ndarray],
        dt: float
    ) -> (np.ndarray, np.ndarray):
        pass


class EulerMethodIntegrator(NumericalIntegrator):
    def __init__(
        self,
        x: np.ndarray,
        v: np.ndarray,
        accel_func: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        self.x = x.copy()
        self.v = v.copy()

    def step(
        self,
        accel_func: Callable[[np.ndarray], np.ndarray],
        dt: float
    ) -> (np.ndarray, np.ndarray):
        self.x += self.v * dt
        self.v += accel_func(self.v) * dt
        return self.x, self.v


class HeunsMethodIntegrator(NumericalIntegrator):
    def __init__(
        self,
        x: np.ndarray,
        v: np.ndarray,
        accel_func: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        self.x = x.copy()
        self.v = v.copy()

    def step(
        self,
        accel_func: Callable[[np.ndarray], np.ndarray],
        dt: float
    ) -> (np.ndarray, np.ndarray):
        accel = accel_func(self.v)
        v_pred = self.v + accel_func(self.v) * dt
        accel_pred = accel_func(v_pred)

        self.x += (self.v + v_pred) / 2.0 * dt
        self.v += (accel + accel_pred) / 2.0 * dt

        return self.x, self.v


class BeemansAlgorithmIntegrator(NumericalIntegrator):
    def __init__(
        self,
        x: np.ndarray,
        v: np.ndarray,
        accel_func: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        self.x = x.copy()
        self.v = v.copy()
        self.accel_prev = accel_func(self.v)

    def step(
        self,
        accel_func: Callable[[np.ndarray], np.ndarray],
        dt: float
    ) -> (np.ndarray, np.ndarray):
        accel_curr = accel_func(self.v)
        v_next = self.v + (3.0 * accel_curr - self.accel_prev) / 2.0 * dt
        accel_next = accel_func(v_next)

        x_next = self.x + self.v * dt + \
            (accel_next + 2.0 * accel_curr) / 6.0 * dt * dt
        v_next = self.v + (5.0 * accel_next + 8.0 *
                           accel_curr - self.accel_prev) / 12.0 * dt
        accel_next = accel_func(v_next)

        self.x = x_next
        self.v = v_next
        self.accel_prev = accel_next

        return self.x, self.v


class RungeKuttaMethodIntegrator(NumericalIntegrator):
    def __init__(
        self,
        x: np.ndarray,
        v: np.ndarray,
        accel_func: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        self.x = x.copy()
        self.v = v.copy()

    def step(
        self,
        accel_func: Callable[[np.ndarray], np.ndarray],
        dt: float
    ) -> (np.ndarray, np.ndarray):
        k1 = accel_func(self.v)
        v_k1 = self.v + k1 * dt / 2.0
        k2 = accel_func(v_k1)
        v_k2 = self.v + k2 * dt / 2.0
        k3 = accel_func(v_k2)
        v_k3 = self.v + k3 * dt
        k4 = accel_func(v_k3)

        self.x += (self.v + 2.0 * v_k1 + 2.0 * v_k2 + v_k3) / 6.0 * dt
        self.v += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0 * dt

        return self.x, self.v
