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