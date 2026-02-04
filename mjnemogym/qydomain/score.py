# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import logging

_logger = logging.getLogger("mjnemogym.qydomain")

# ==========================================
# Shared Helper Functions
# ==========================================

def levenshtein_distance(A: str, B: str) -> int:
    """
    Calculates the Levenshtein distance between two sequences A and B using Dynamic Programming.
    Used to replace difflib for similarity checks.
    """
    N, M = len(A), len(B)
    # Create an array of size (N+1)x(M+1)
    dp = [[0 for i in range(M + 1)] for j in range(N + 1)]

    # Base Case: When N = 0
    for j in range(M + 1):
        dp[0][j] = j
    # Base Case: When M = 0
    for i in range(N + 1):
        dp[i][0] = i

    # Transitions
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if A[i - 1] == B[j - 1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],   # Insertion
                    dp[i][j-1],   # Deletion
                    dp[i-1][j-1]  # Replacement
                )

    return dp[N][M]

def extract_answer(llm_answer: str) -> str:
    """
    Extracts the answer part from a string following the pattern '... --- answer --- ...'.
    """
    pattern = r'.* --- (.*?) --- .*'
    match = re.search(pattern, llm_answer)
    return match.group(1) if match else llm_answer

# ==========================================
# Evaluator 1: Plot Unscrambling
# ==========================================

def extract_plot_summary(text: str) -> str:
    pattern = r'<PLOT_SUMMARY>(.*)</PLOT_SUMMARY>'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        pattern = r'<PLOT_SUMMARY>(.*)'
        match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else text

def plot_unscrambling_process_results(ground_truth: str, llm_answer: str, debug=False) -> float:
    """
    Evaluates how well the LLM ordered sentences compared to the ground truth.
    Uses Levenshtein distance on the sentence indices.
    """
    # Extract relevant text
    llm_answer = extract_plot_summary(llm_answer)

    # Split into sentences
    gt_sentences = [s.strip() for s in ground_truth.split('.')]
    ans_sentences = [
        s.strip() for s in llm_answer.split('.')
        if s.strip() != '</PLOT_SUMMARY>' and s.strip() != '**End of Plot Summary**'
    ]

    # Filter empty sentences
    gt_sentences = [s for s in gt_sentences if s]
    ans_sentences = [s for s in ans_sentences if s]

    ans_ordering = []

    # Map ground truth sentences to the answer sentences
    for x in gt_sentences:
        if not ans_sentences:
            break

        # Replacement for difflib.get_close_matches:
        # Find the sentence in 'ans_sentences' with the smallest Levenshtein distance to 'x'
        best_match = None
        min_dist = float('inf')

        for candidate in ans_sentences:
            dist = levenshtein_distance(x, candidate)
            # Find the closest match (simulating cutoff=0.0 logic)
            if dist < min_dist:
                min_dist = dist
                best_match = candidate

        if best_match:
            try:
                ans_ordering.append(ans_sentences.index(best_match))
            except ValueError:
                pass

    n_sentences_gt = len(gt_sentences)
    if n_sentences_gt == 0:
        return 0.0

    # Calculate edit distance between the expected index order (0, 1, 2...) and actual found order
    raw_distance = levenshtein_distance(list(range(len(gt_sentences))), ans_ordering)
    score = 1 - (raw_distance / n_sentences_gt)

    if debug and score < 1:
        print(f'[DEBUG-PLOT] INCORRECT Score: {score}')
        print(f'[DEBUG-PLOT] GT Sentences: {gt_sentences}')
        print(f'[DEBUG-PLOT] Ans Sentences: {ans_sentences}')

    return max(0.0, score)

# ==========================================
# Evaluator 2: Connections Puzzle
# ==========================================

def last_boxed_only_string(string: str):
    """Parses LaTeX style \\boxed{content}."""
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]
    return retval

def remove_boxed(s: str):
    left = "\\boxed{"
    try:
        if s[: len(left)] == left and s[-1] == "}":
            return s[len(left) : -1]
        return None
    except Exception:
        return None

def group_words(words: list):
    """Groups a list of words into sets of 4."""
    groups = [set()]
    words = [w.strip().lower() for w in words]
    for word in words:
        if len(groups[-1]) == 4:
            groups.append(set())
        groups[-1].add(word)
    return groups

def connections_process_results(ground_truth: str, llm_answer: str, debug=False) -> float:
    """Evaluator for newer puzzles (looks for <solution> tags or boxed text)."""

    # Try to find content inside <solution> tags
    solution_matches = re.findall(r'<solution>(.*?)<\/solution>', llm_answer)
    if not solution_matches:
        solution_matches = re.findall(r'<solution>(.*?)<\/solution>', llm_answer.replace('\n', ''))
    if not solution_matches:
        # Check for malformed closing tags scenarios
        solution_matches = re.findall(r'</solution>(.*?)<\/solution>', llm_answer)

    ground_truth_words = ground_truth.split(',')

    # Fallback to \boxed format if no xml tags found
    if len(solution_matches) == 0 and '\\boxed' in llm_answer:
        boxed = last_boxed_only_string(llm_answer)
        if boxed:
            no_box = remove_boxed(boxed)
            if no_box:
                # Clean up latex syntax
                clean_text = no_box.replace('\\text{', '').replace('}', '').replace('\\', '')
                solution_matches = [clean_text]

    # Clean newlines from matches
    solution_matches = [match.replace('\n', '') for match in solution_matches]

    if len(solution_matches) == 0:
        if debug:
            print('[DEBUG-CONN] No solution text found.')
        return 0

    # Handle multiple matches or single match
    if len(solution_matches) > 1:
        if debug:
            print('[DEBUG-CONN] Multiple solution texts found. Combining from last.')
        all_words = []
        num_words = len(ground_truth_words)
        for match in solution_matches:
            all_words.extend(match.split(','))
        solution_words = all_words[-num_words:]
    else:
        solution_words = solution_matches[-1].split(',')

    # Compare Groups
    llm_groups = group_words(solution_words)
    ground_truth_groups = group_words(ground_truth_words)

    correct_groups = 0
    for llm_group in llm_groups:
        if llm_group in ground_truth_groups:
            correct_groups += 1

    if len(ground_truth_groups) == 0:
        return 0

    score = correct_groups / len(ground_truth_groups)

    if debug and score < 1:
        print(f'[DEBUG-CONN] Incorrect. Score: {score}')
        print(f'GT Groups: {sorted([sorted(list(g)) for g in ground_truth_groups])}')
        print(f'LLM Groups: {sorted([sorted(list(g)) for g in llm_groups])}')

    return score

# ==========================================
# Evaluator 3: Typos / Exact Match
# ==========================================

def typos_process_results(ground_truth: str, llm_answer: str, debug=False) -> int:
    """
    Checks if the ground truth is present in the LLM answer.
    """
    parsed_answer = None

    # Priority 1: Extract from <solution> tags
    solution_matches = re.findall(r'<solution>(.*?)</solution>', llm_answer)
    if len(solution_matches) > 0:
        match = solution_matches[-1]
        parsed_answer = match
    else:
        # Priority 2: Clean tags and use separator pattern extraction
        parsed_answer = llm_answer.replace('<solution>', '').replace('</solution>', '')
        parsed_answer = extract_answer(parsed_answer)

    # Clean up whitespace/newlines
    parsed_answer = ' '.join(list(filter(None, parsed_answer.strip().split('\n'))))

    # Core Logic: Check for substring inclusion
    if int(ground_truth in parsed_answer):
        return 1

    # Simplified Debug Logic (No difflib)
    score = 0
    if debug and score == 0:
        print('[DEBUG-TYPO] INCORRECT')
        print(f'GT  : {ground_truth}')
        print(f'PRED: {parsed_answer}')

    return score


# ==========================================
# API Wrappers for mjnemogym integration
# ==========================================

def typos_score_fn(model_output: str, extra_info: dict) -> float:
    """Score function for typos domain."""
    label = extra_info["label"]
    return float(typos_process_results(label, model_output))


def connections_score_fn(model_output: str, extra_info: dict) -> float:
    """Score function for connections domain."""
    label = extra_info["label"]
    return float(connections_process_results(label, model_output))


def unscrambling_score_fn(model_output: str, extra_info: dict) -> float:
    """Score function for unscrambling domain."""
    label = extra_info["label"]
    return float(plot_unscrambling_process_results(label, model_output))
