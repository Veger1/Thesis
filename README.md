Repository contains code used in Master's thesis: Reducing spacecraft microvibrations through enhanced reaction wheel array control allocation techniques

The main scripts are calc.py and repeat_solver.py.
    Specify the cost function and solver in calc.py and run it to get the results.
    The solving is done with code in repeat_solver.py.

Keep in mind that a full slew is 800seconds and corresponds to 8004 points in time.
Approximate conversion:
    16 x 500 points = 16 x 50.0 seconds
    8 x 1000 points = 8 x 100.0 seconds
Ideal conversion for 4 part slew:
    4 x 2001 points = 4 x 200.0 seconds
