**You should be able to answer:**
- What are neighboring datasets?
- What do ε and δ mean (in one sentence)?
- What is sensitivity and how does it set the noise?
- What are Laplace/Gaussian mechanisms?
- What is post‑processing? What is composition?


# 🛠️ Chapter 2

Chapter 2 is about the **tools** that make Differential Privacy (DP) actually work.  
Think of this as your **DP toolbox**.

---

## 1. Global Sensitivity

👉 **Definition:**  
Sensitivity = the **maximum change** in a function’s output if you change or remove just **one person’s data**.

👉 **Why it matters:**  
If one person can swing the result a lot, you need **more noise** to hide them.  
If one person barely affects it, you can add **less noise**.

**Examples:**
- **Counting people:** sensitivity = 1 (one person can change the count by at most 1).  
- **Average rating (0–5 stars, n people):** sensitivity = 5/n (one person changing from 0 → 5 changes the mean by at most 5/n).  

---

## 2. Laplace Mechanism (Pure DP, δ=0)

👉 **Definition:**  
Add noise from a **Laplace distribution** with scale:

\[
b = \frac{\text{sensitivity}}{\varepsilon}
\]

👉 **Meaning in practice:**  
- Compute your answer (e.g., “42 people like pizza”).  
- Add Laplace noise.  
- Output might be 41.2 or 43.7.  

**Takeaway:** Smaller ε → bigger noise → stronger privacy.

---

## 3. Gaussian Mechanism (Approximate DP, δ>0)

👉 **Definition:**  
Add noise from a **Gaussian (bell curve) distribution** with standard deviation:

\[
\sigma = \frac{\sqrt{2 \ln(1.25/\delta)} \cdot \text{sensitivity}}{\varepsilon}
\]

👉 **Meaning in practice:**  
- Similar to Laplace, but with bell-curve noise instead of spiky noise.  
- Allows a tiny failure probability δ.  
- Friendlier for many repeated queries (composition).  

---

## 4. Post-Processing Invariance

👉 **Definition:**  
If the output of a mechanism is DP, then **any further computation** on it is still DP.

👉 **Why it matters:**  
Once noise is added, you can:  
- Round it  
- Plot it  
- Feed it into another model  

…and the DP guarantee remains intact.  
**You can’t unblur it.**

---

## 5. Composition

👉 **Definition:**  
Each DP query spends privacy budget. Running multiple queries **adds up** the cost.

- **Sequential composition (same users):**  
  If M1 is (ε₁, δ₁)-DP and M2 is (ε₂, δ₂)-DP, together they are  
  (ε₁ + ε₂, δ₁ + δ₂)-DP.

- **Parallel composition (disjoint users):**  
  If queries touch different people, the cost is just the **max**, not the sum.

👉 **Why it matters:**  
- You have a **privacy wallet**.  
- Each query spends some ε (and δ).  
- Too many queries → budget runs out.  
- Forces you to plan carefully.

---

## 6. Quick Analogy

- **Sensitivity** = ruler: how much one person can change the answer.  
- **Laplace / Gaussian** = noise machines to blur the answer.  
- **Post-processing** = once blurred, it stays blurred.  
- **Composition** = wallet: every query spends a bit of your budget.

---

✅ **Chapter 2 takeaway:**  
DP gives you tools to **measure impact (sensitivity)**, **add just enough noise (Laplace/Gaussian)**, and **track your budget (composition)**, while guaranteeing that once noise is added, privacy protection sticks (**post-processing**).
