Repository contains code used in Master's thesis: Reducing spacecraft microvibrations through enhanced reaction wheel array control allocation techniques

The main scripts are ocp_solver and ocp_repeat_solver.
    - ocp_solver.py: Solves the optimal control problem for N points in time.
    - ocp_repeat_solver.py: Solves the optimal control problem for N points in time,
        repeating the process for M iterations.
    - Post-processing.py: Plots saved data from the solver scripts.

Keep in mind that a full slew is 800seconds and corresponds to 8004 points in time.
Approximate conversion:
    16 x 500 points = 16 x 50.0 seconds
    8 x 1000 points = 8 x 100.0 seconds
Ideal conversion for 4 part slew:
    4 x 2001 points = 4 x 200.0 seconds


Omega_square used real-time control allocation. Every step it calculates alpha that results in lowest sum of squares
of reaction wheel speeds. It essentially removes any  torque/speed that is in the nullspace of the control allocation matrix.
Offset can be added by adding a constant to alpha after calculation or adding initial conditions which are not in the nullspace.