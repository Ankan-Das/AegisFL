'''
1. Global Sensitivity

👉 Definition:
For a function f that looks at a dataset, sensitivity = the maximum change in f's output if you change just one person's data.

👉 What it really means:
How much impact can a single person have?
If it's big → you need lots of noise to hide them.
If it's small → you can get away with less noise.
'''


import numpy as np

# Example: count how many people like pizza
def count_pizza(data: list[int]) -> int:
    return sum(data)

# Two neighbouring datasets: differ by one person
D1 = [1, 0, 1, 1, 0]
D2 = [1, 0, 1, 1, 1]

print("f(D1) =", count_pizza(D1))
print("f(D2) =", count_pizza(D2))

# Sensitivity is the maximum difference between the outputs of two neighbouring datasets
sensitivity = abs(count_pizza(D1) - count_pizza(D2))
print("Sensitivity =", sensitivity)

'''
Sensitivity is a measure of how much the output of a function changes when the input changes by one.
It is used to quantify the privacy loss of a differentially private mechanism.

Here, Sensitivity = 1 (one person can change the count by 1).
'''