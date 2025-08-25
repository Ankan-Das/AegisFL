'''
👉 Definition:
Release f(D) plus Gaussian (normal) noise with standard deviation

👉 What it really means:
Same idea as Laplace, but use bell-curve noise instead.
It introduces a tiny failure probability δ (like 1 in a million chance privacy is broken).
We allow this because Gaussian noise is friendlier when you compose many queries (it doesn't blow up as badly).
'''
import numpy as np

def gaussian_noise(sigma):
    return np.random.normal(0, sigma)

true_count = 42
sensitivity = 1
epsilon = 1.0
delta = 1e-5

sigma = np.sqrt(2*np.log(1.25/delta)) * (sensitivity/epsilon)

noisy_answers = [true_count + gaussian_noise(sigma) for _ in range(10)]
print("Noisy counts:", [round(x,2) for x in noisy_answers])


'''
What you'll see: Also noisy answers, but noise looks “smoother” (Laplace sometimes jumps more).

👉 Gaussian is friendlier when we repeat queries many times.
'''