# QY Domain Scoring Functions

This module provides scoring functions for three language-based evaluation tasks:
1. **Typos** - Spelling correction verification
2. **Connections** - Word grouping puzzle evaluation
3. **Unscrambling** - Plot sentence ordering evaluation

## Data Format

```python
{
    "data_source": "typos" | "connections" | "unscrambling",
    "extra_info": {
        "label": "ground truth text"
    }
}
```

---

## 1. Typos Evaluator (`typos_process_results`)

### Description
Evaluates whether the LLM correctly spelled a word by checking if the ground truth appears as a substring in the parsed answer.

### Algorithm
1. Extract answer from `<solution>...</solution>` tags (priority 1) or `--- answer ---` pattern (priority 2)
2. Check if ground truth is a substring of the parsed answer

### Scoring Formula

$$
\text{score} = \mathbb{1}[\text{ground\_truth} \subseteq \text{parsed\_answer}] =
\begin{cases}
1 & \text{if ground\_truth} \in \text{parsed\_answer} \\
0 & \text{otherwise}
\end{cases}
$$

### Properties
- **Range**: $\{0, 1\}$ (binary)
- **Guaranteed bounds**: Yes, always 0 or 1

### Example

| Ground Truth | LLM Output | Score |
|--------------|------------|-------|
| `extraordinary` | `<solution>extraordinary</solution>` | 1 |
| `extraordinary` | `<solution>extraordinry</solution>` | 0 |
| `hello` | `The answer is --- hello --- done` | 1 |
| `hello` | `The word is hello.` | 1 |

```python
>>> typos_process_results("extraordinary", "<solution>extraordinary</solution>")
1
>>> typos_process_results("extraordinary", "<solution>extraordinry</solution>")
0
```

---

## 2. Connections Evaluator (`connections_process_results`)

### Description
Evaluates word grouping puzzles where the LLM must identify groups of 4 related words from a set of 16 words.

### Algorithm
1. Extract comma-separated words from `<solution>...</solution>` tags or `\boxed{...}` format
2. Group words into sets of 4 (sequential grouping)
3. Compare LLM groups against ground truth groups (case-insensitive, order-independent within groups)

### Scoring Formula

Let $G = \{G_1, G_2, ..., G_k\}$ be the ground truth groups and $L = \{L_1, L_2, ..., L_m\}$ be the LLM groups.

$$
\text{score} = \frac{|\{L_i \in L : L_i \in G\}|}{|G|} = \frac{\text{number of correct groups}}{\text{total ground truth groups}}
$$

Where group equality is defined as set equality (case-insensitive):
$$
L_i = G_j \iff \{\text{lower}(w) : w \in L_i\} = \{\text{lower}(w) : w \in G_j\}
$$

### Properties
- **Range**: $[0, 1]$ where score $\in \{0, \frac{1}{k}, \frac{2}{k}, ..., 1\}$ for $k$ groups
- **Guaranteed bounds**: Yes, $0 \leq \text{score} \leq 1$

### Example

**Ground Truth**: `Apple,Banana,Pear,Grape,Red,Blue,Green,Yellow` (2 groups of 4)
- Group 1: `{apple, banana, pear, grape}`
- Group 2: `{red, blue, green, yellow}`

| LLM Output | Groups Formed | Correct | Score |
|------------|---------------|---------|-------|
| `<solution>Apple,Banana,Pear,Grape,Red,Blue,Green,Yellow</solution>` | `{apple,banana,pear,grape}`, `{red,blue,green,yellow}` | 2/2 | 1.0 |
| `<solution>Apple,Banana,Pear,Orange,Red,Blue,Green,Yellow</solution>` | `{apple,banana,pear,orange}`, `{red,blue,green,yellow}` | 1/2 | 0.5 |
| `<solution>Red,Apple,Blue,Banana,Green,Pear,Yellow,Grape</solution>` | `{red,apple,blue,banana}`, `{green,pear,yellow,grape}` | 0/2 | 0.0 |

```python
>>> connections_process_results(
...     "Apple,Banana,Pear,Grape,Red,Blue,Green,Yellow",
...     "<solution>Apple,Banana,Pear,Grape,Red,Blue,Green,Yellow</solution>"
... )
1.0
>>> connections_process_results(
...     "Apple,Banana,Pear,Grape,Red,Blue,Green,Yellow",
...     "<solution>Apple,Banana,Pear,Orange,Red,Blue,Green,Yellow</solution>"
... )
0.5
```

---

## 3. Unscrambling Evaluator (`plot_unscrambling_process_results`)

### Description
Evaluates how well the LLM reordered scrambled plot sentences. The score measures how close the predicted ordering is to the correct ordering using Levenshtein distance on sentence indices.

### Algorithm

1. **Extract**: Get content from `<PLOT_SUMMARY>...</PLOT_SUMMARY>` tags
2. **Split**: Split both ground truth and answer into sentences by `.`
3. **Match**: For each ground truth sentence $g_i$, find the best matching answer sentence $a_j$ using Levenshtein distance on the text:
   $$j^* = \arg\min_{j} d_{\text{Lev}}(g_i, a_j)$$
4. **Build ordering**: Record indices to form $O = [o_0, o_1, ..., o_{k-1}]$ where $o_i = j^*$
5. **Compare**: Calculate Levenshtein distance between expected ordering $E = [0, 1, 2, ..., n-1]$ and actual ordering $O$

### Scoring Formula

Let:
- $n$ = number of sentences in ground truth
- $E = [0, 1, 2, ..., n-1]$ = expected index ordering
- $O = [o_0, o_1, ..., o_{k-1}]$ = actual index ordering from LLM (where $k \leq n$)
- $d_{\text{Lev}}(E, O)$ = Levenshtein distance between the two index sequences

$$
\text{score} = \max\left(0, 1 - \frac{d_{\text{Lev}}(E, O)}{n}\right)
$$

### Properties
- **Range**: $[0, 1]$
- **Guaranteed bounds**: Yes
  - Upper bound: When $O = E$, $d_{\text{Lev}}(E, O) = 0$, so score $= 1$
  - Lower bound: Clamped to 0 via `max(0, ...)`
  - Theoretical minimum before clamping: Since $|E| = n$ and $|O| \leq n$, we have $d_{\text{Lev}}(E, O) \leq n$, so score $\geq 0$

### Levenshtein Distance on Index Sequences

The Levenshtein distance counts the minimum number of single-element edits (insertions, deletions, substitutions) needed to transform one sequence into another.

**Example**: $E = [0, 1, 2]$, $O = [0, 2, 1]$

Operations to transform $O \to E$:
- Replace index 1: $2 \to 1$
- Replace index 2: $1 \to 2$

$d_{\text{Lev}}([0,1,2], [0,2,1]) = 2$

Score $= 1 - \frac{2}{3} = 0.333$

### Example

**Ground Truth**: `"The hero wakes up. He fights the dragon. He wins the gold."`
- Sentences: $G = [\text{"The hero wakes up"}, \text{"He fights the dragon"}, \text{"He wins the gold"}]$
- Expected ordering: $E = [0, 1, 2]$

| LLM Output | Matched Ordering $O$ | $d_{\text{Lev}}(E, O)$ | Score |
|------------|---------------------|------------------------|-------|
| `<PLOT_SUMMARY>The hero wakes up. He fights the dragon. He wins the gold.</PLOT_SUMMARY>` | $[0, 1, 2]$ | 0 | 1.0 |
| `<PLOT_SUMMARY>The hero wakes up. He wins the gold. He fights the dragon.</PLOT_SUMMARY>` | $[0, 2, 1]$ | 2 | 0.33 |
| `<PLOT_SUMMARY>He wins the gold. He fights the dragon. The hero wakes up.</PLOT_SUMMARY>` | $[2, 1, 0]$ | 2 | 0.33 |
| `<PLOT_SUMMARY>He fights the dragon. The hero wakes up. He wins the gold.</PLOT_SUMMARY>` | $[1, 0, 2]$ | 2 | 0.33 |

```python
>>> plot_unscrambling_process_results(
...     "The hero wakes up. He fights the dragon. He wins the gold.",
...     "<PLOT_SUMMARY>The hero wakes up. He fights the dragon. He wins the gold.</PLOT_SUMMARY>"
... )
1.0
>>> plot_unscrambling_process_results(
...     "The hero wakes up. He fights the dragon. He wins the gold.",
...     "<PLOT_SUMMARY>The hero wakes up. He wins the gold. He fights the dragon.</PLOT_SUMMARY>"
... )
0.3333333333333333
```

### Detailed Walkthrough

**Input**:
- Ground truth: `"A. B. C."` → $G = [\text{"A"}, \text{"B"}, \text{"C"}]$, $n = 3$
- LLM answer: `"<PLOT_SUMMARY>B. A. C.</PLOT_SUMMARY>"` → $A = [\text{"B"}, \text{"A"}, \text{"C"}]$

**Step 3 - Matching**:
| GT Sentence | Best Match in $A$ | Index in $A$ |
|-------------|-------------------|--------------|
| "A" | "A" (dist=0) | 1 |
| "B" | "B" (dist=0) | 0 |
| "C" | "C" (dist=0) | 2 |

**Step 4**: $O = [1, 0, 2]$

**Step 5**:
- $E = [0, 1, 2]$
- $d_{\text{Lev}}([0,1,2], [1,0,2]) = 2$ (swap first two elements)

**Score**: $1 - \frac{2}{3} = 0.333$

---

## API Usage

```python
from mjnemogym.qydomain import (
    typos_score_fn,
    connections_score_fn,
    unscrambling_score_fn,
)

# Typos
score = typos_score_fn(
    model_output="<solution>extraordinary</solution>",
    extra_info={"label": "extraordinary"}
)  # Returns 1.0

# Connections
score = connections_score_fn(
    model_output="<solution>a,b,c,d,e,f,g,h</solution>",
    extra_info={"label": "a,b,c,d,e,f,g,h"}
)  # Returns 1.0

# Unscrambling
score = unscrambling_score_fn(
    model_output="<PLOT_SUMMARY>First. Second. Third.</PLOT_SUMMARY>",
    extra_info={"label": "First. Second. Third."}
)  # Returns 1.0
```

## Summary Table

| Evaluator | Score Range | Score Type | Key Metric |
|-----------|-------------|------------|------------|
| Typos | $\{0, 1\}$ | Binary | Substring match |
| Connections | $[0, 1]$ | Discrete ($\frac{k}{n}$) | Group set equality |
| Unscrambling | $[0, 1]$ | Continuous | Levenshtein distance on indices |
