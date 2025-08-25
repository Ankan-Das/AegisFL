'''
👉 Idea: “Each DP query eats privacy budget. They add up.”
'''

# Suppose each query uses epsilon=0.5
eps_per_query = 0.5
num_queries = 4

total_epsilon = eps_per_query * num_queries
print("Total privacy cost:", total_epsilon)

'''
Output:

Total privacy cost: 2.0

👉 If ε=0.5 is “pretty private”, then 4 queries at that level = 2.0 total privacy cost (less private).
'''

'''
1. What is a “query”?

A query = any computation you want to do on the dataset.

Examples:

“How many people like pizza?”

“What’s the average age?”

“Train a model and give me its accuracy?”

Each time you ask something of the data, that’s a query.

2. Why is a query sensitive?

Because if you don’t protect it, one person’s data can change the answer.
Differential Privacy (DP) fixes this by adding noise to the query result so no one can tell if that person was included.

3. Why does each query eat privacy budget?

Think of it like this:

When you release one noisy answer, it hides any one person pretty well (ε small).

But if you release many noisy answers, someone could start piecing them together to reduce the uncertainty.

Example: If I ask “how many like pizza?” 10 times and you add noise each time, I could average the results and cancel out the noise.

That would leak more about individuals.

So, the more queries you run, the more total privacy “spent”.

That “spending” is measured by ε (and δ).

4. How does it add up?

The composition rule says:

If the first query is (ε1,δ1)-DP,

and the second is (𝜀2,𝛿2)-DP,

then both together are (𝜀1+𝜀2, 𝛿1+𝛿2)-DP.

👉 In words: the privacy risks add.
It's like stacking layers of glass: each one is see-through a little bit; stack too many and people can see through clearly.

5. Analogy: Privacy Wallet 💳

Every person starts with a “privacy wallet” containing budget ε.

Each query = a transaction, spends some budget.

When the budget is gone → too many queries = can't guarantee privacy anymore.

You (the system designer) must choose how to allocate the budget:

Fewer queries → less total spend, more privacy.

More queries → more total spend, weaker privacy.

6. Mini demo in Python
# Privacy wallet example
eps_budget = 2.0     # total wallet
eps_per_query = 0.5  # cost per query

wallet = eps_budget
for i in range(1, 6):
    if wallet >= eps_per_query:
        wallet -= eps_per_query
        print(f"Query {i}: answered (spent {eps_per_query}), remaining = {wallet}")
    else:
        print(f"Query {i}: NOT allowed, budget exhausted!")


Output:

Query 1: answered (spent 0.5), remaining = 1.5
Query 2: answered (spent 0.5), remaining = 1.0
Query 3: answered (spent 0.5), remaining = 0.5
Query 4: answered (spent 0.5), remaining = 0.0
Query 5: NOT allowed, budget exhausted!


👉 You only got 4 queries before the budget ran out.

✅ So:

Query = any computation answered with DP.

Each query leaks a little.

Leaks add up → privacy budget (ε) decreases.

This forces you to design systems carefully (decide which queries are worth privacy spend).
'''