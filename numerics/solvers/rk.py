from scipy.integrate import solve_ivp

def solve_ode(rhs, r_span, u0):
    """
    Wrapper for rk4 integrator. 
    inputs: 
    rhs - ODE to be solved
    r_span - array - radial datapoints
    u0 - array - initial velocity profile

    -----

    outputs:
    solution - array - velocity profile out to limits of integration
    """
    return solve_ivp(rhs, r_span, u0, method='RK45', rtol = 1e-8, atol=1e-10, dense_output = True)

