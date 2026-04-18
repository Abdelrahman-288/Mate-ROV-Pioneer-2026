import numpy as np

def distance(a, b):
    return np.sqrt((a[0]-b[0])**2 +
                   (a[1]-b[1])**2 +
                   (a[2]-b[2])**2)

# Example: replace with real extracted points
T1 = (0, 0, 0)
T8 = (10, 2, 1)

length = distance(T1, T8)

print("Coral Ridge Length:", length, "units")