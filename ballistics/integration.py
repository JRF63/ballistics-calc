from abc import ABC, abstractmethod
from typing import Callable
from collections import deque

import numpy as np
from scipy.integrate import OdeSolver, DenseOutput
from scipy.interpolate import make_interp_spline


class SplineDenseOutput(DenseOutput):
    def __init__(self, t_old, t, hist):
        super().__init__(t_old, t)
        self.hist = hist

    def _call_impl(self, t):
        if len(self.hist) < 4:
            spline = make_interp_spline(*zip(*self.hist), k=1)
        else:
            spline = make_interp_spline(*zip(*self.hist))

        if t.ndim == 0:
            return spline(t)
        else:
            ret = np.array([spline(x) for x in t])
            return ret.T


class CustomOdeSolver(OdeSolver):
    def __init__(self, fun, t0, y0, t_bound, h, **extraneous):
        super().__init__(fun, t0, y0, t_bound, vectorized=False, support_complex=True)
        self.h = h
        self.hist = deque(maxlen=4)
        self.hist.append((self.t, self.y.copy()))

    def _step_impl(self):
        self.hist.append((self.t, self.y.copy()))
        return True, None

    def _dense_output_impl(self):
        return SplineDenseOutput(self.t_old, self.t, self.hist)


class EulerMethod(CustomOdeSolver):
    def __init__(self, fun, t0, y0, t_bound, h=1.0/60.0, **extraneous):
        super().__init__(fun, t0, y0, t_bound, h, **extraneous)

    def _step_impl(self):
        t_new = self.t + self.h
        if t_new - self.t_bound > 0.0:
            t_new = self.t_bound
        h = t_new - self.t

        self.y += self.fun(self.t, self.y) * h
        self.t = t_new

        return super()._step_impl()


class TwoStepAdamsBashforth(CustomOdeSolver):
    def __init__(self, fun, t0, y0, t_bound, h=1.0/60.0, **extraneous):
        super().__init__(fun, t0, y0, t_bound, h, **extraneous)
        self.derivative_old = self.fun(self.t, self.y)

    def _step_impl(self):
        t_new = self.t + self.h
        if t_new - self.t_bound > 0.0:
            t_new = self.t_bound
        h = t_new - self.t

        derivative = self.fun(self.t, self.y)
        self.y += (3.0 * derivative - self.derivative_old) / 2.0 * h
        self.t = t_new
        self.derivative_old = derivative

        return super()._step_impl()


class HeunsMethod(CustomOdeSolver):
    def __init__(self, fun, t0, y0, t_bound, h=1.0/60.0, **extraneous):
        super().__init__(fun, t0, y0, t_bound, h, **extraneous)

    def _step_impl(self):
        t_new = self.t + self.h
        if t_new - self.t_bound > 0.0:
            t_new = self.t_bound
        h = t_new - self.t

        derivative = self.fun(self.t, self.y)
        y_pred = self.y + derivative * h
        derivative_pred = self.fun(t_new, y_pred)

        self.y += (derivative + derivative_pred) / 2.0 * h
        self.t = t_new

        return super()._step_impl()


class BeemansAlgorithm(CustomOdeSolver):
    def __init__(self, fun, t0, y0, t_bound, h=1.0/60.0, **extraneous):
        super().__init__(fun, t0, y0, t_bound, h, **extraneous)
        self.derivative_old = self.fun(self.t, self.y)

    def _step_impl(self):
        t_new = self.t + self.h
        if t_new - self.t_bound > 0.0:
            t_new = self.t_bound
        h = t_new - self.t

        derivative = self.fun(self.t, self.y)
        y_new = self.y + (3.0 * derivative - self.derivative_old) / 2.0 * h

        derivative_new = self.fun(t_new, y_new)
        y_new[:3] = self.y[:3] + self.y[3:] * h + \
            (derivative_new + 2.0 * derivative)[:3] / 6.0 * h * h
        y_new[3:] = self.y[3:] + (5.0 * derivative_new + 8.0 *
                                  derivative - self.derivative_old)[3:] / 12.0 * h

        self.y = y_new
        self.t = t_new
        self.derivative_old = derivative

        return super()._step_impl()


class RungeKuttaMethod(CustomOdeSolver):
    def __init__(self, fun, t0, y0, t_bound, h=1.0/60.0, **extraneous):
        super().__init__(fun, t0, y0, t_bound, h, **extraneous)

    def _step_impl(self):
        t_new = self.t + self.h
        if t_new - self.t_bound > 0.0:
            t_new = self.t_bound
        h = t_new - self.t

        k1 = self.fun(self.t, self.y)
        k2 = self.fun(self.t + h / 2.0, self.y + k1 * h / 2.0)
        k3 = self.fun(self.t + h / 2.0, self.y + k2 * h / 2.0)
        k4 = self.fun(self.t + h, self.y + k3 * h)

        self.y += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0 * h
        self.t = t_new

        return super()._step_impl()
