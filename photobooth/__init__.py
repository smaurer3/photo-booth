import os

# The keyer's Numba threads use OpenMP, whose threads busy-wait between parallel loops by
# default. At 30 preview frames a second they never get to sleep, so every core looks busy
# (measured: 6.8 cores busy for 2.7 cores of work). Passive threads sleep instead; the output
# is identical. It must be set before Numba starts its threads, hence here.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
