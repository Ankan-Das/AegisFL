'''
2. Laplace Mechanism (δ = 0, “pure DP”)

👉 Definition:
Release f(D) plus noise from a Laplace distribution with scale:
b=sensitivity/ε


👉 What it really means:
Take your true answer (e.g. “42 people liked the movie”) and add random noise (like rolling dice).
The amount of noise depends on:

Sensitivity (how much one person can swing the answer)

ε (privacy budget; smaller ε = stronger privacy = more noise)
'''
import numpy as np

def laplace_noise(scale):
    return np.random.laplace(0, scale)

true_count = 42  # say 42 people like pizza
sensitivity = 1
epsilon = 1.0
scale = sensitivity / epsilon

noisy_answers = [true_count + laplace_noise(scale) for _ in range(10)]
print(noisy_answers)
print("Noisy counts:", [round(x,2) for x in noisy_answers])

'''
👉 One person's data is “hidden” inside that fuzz.
'''