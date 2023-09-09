import contextlib
import numpy as np
from matplotlib import pyplot as plt

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

w_inf = []
def EulerIntegrate(controller, f, B, Bw, xstar, ustar, xinit, t_max = 10, dt = 0.05, with_tracking = False, distb = None):
    t = np.arange(0, t_max, dt)
    trace = []
    u = []
    xcurr = xinit
    trace.append(xcurr)

    for i in range(len(t)):
        w = distb[i,:,:]
        if with_tracking:
            xe = xcurr - xstar[i]
        ui = controller(xcurr, xe, ustar[i]) if with_tracking else ustar[i]
        if with_tracking:
            # print(xcurr.reshape(-1), xstar[i].reshape(-1), ui.reshape(-1))
            pass

        dx = f(xcurr) + B(xcurr).dot(ui) + Bw(xcurr).dot(w) if with_tracking else f(xcurr) + B(xcurr).dot(ui)
        xnext =  xcurr + dx*dt

        trace.append(xnext)
        u.append(ui)
        xcurr = xnext

    return trace,u


