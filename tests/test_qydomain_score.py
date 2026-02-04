#!/usr/bin/env python3
"""
Test script for QY domain scoring (typos, connections, unscrambling).

Tests aligned with original qyscore.py test cases.
"""

import concurrent.futures
import sys
import time

from mjnemogym import verl_compute_score
from mjnemogym.qydomain.score import (
    typos_score_fn,
    connections_score_fn,
    unscrambling_score_fn,
    typos_process_results,
    connections_process_results,
    plot_unscrambling_process_results,
)


def test_typos():
    """Test typos evaluator - aligned with qyscore.py test cases."""
    print('=== Typos Evaluator Tests ===\n', flush=True)

    # Test cases from qyscore.py
    gt_typo = "extraordinary"
    ans_typo_correct = "The correct spelling is <solution>extraordinary</solution>."
    ans_typo_wrong = "The correct spelling is <solution>extraordinry</solution>."

    test_cases = [
        # (model_output, ground_truth, expected_score, description)
        (ans_typo_correct, gt_typo, 1, 'Correct answer (qyscore.py case)'),
        (ans_typo_wrong, gt_typo, 0, 'Wrong answer with typo (qyscore.py case)'),
        ('<solution>extraordinary</solution>', 'extraordinary', 1, 'Exact match in solution tag'),
        ('The answer is <solution>hello</solution>.', 'hello', 1, 'Answer with surrounding text'),
        ('--- hello ---', 'hello', 1, 'Answer in separator pattern'),
        ('The correct spelling is extraordinary.', 'extraordinary', 1, 'Substring match'),
        ('no match here', 'extraordinary', 0, 'No match'),
        ('', 'test', 0, 'Empty output'),
    ]

    passed = 0
    for output, gt, expected, desc in test_cases:
        result = typos_process_results(gt, output)
        status = '✓' if result == expected else '✗'
        if result == expected:
            passed += 1
        print(f'  {status} {desc}: {result} (expected {expected})', flush=True)

    print(f'\nPassed: {passed}/{len(test_cases)}', flush=True)
    return passed == len(test_cases)


def test_connections():
    """Test connections evaluator - aligned with qyscore.py test cases."""
    print('\n=== Connections Evaluator Tests ===\n', flush=True)

    # Test cases from qyscore.py
    gt_conn = "Apple,Banana,Pear,Grape,Red,Blue,Green,Yellow"
    ans_conn_correct = "<solution>Apple, Banana, Pear, Grape, Red, Blue, Green, Yellow</solution>"
    ans_conn_partial = "<solution>Apple, Banana, Pear, Orange, Red, Blue, Green, Yellow</solution>"

    test_cases = [
        # (model_output, ground_truth, expected_score, description)
        (ans_conn_correct, gt_conn, 1.0, 'Perfect match (qyscore.py case)'),
        (ans_conn_partial, gt_conn, 0.5, 'One group wrong - Orange vs Grape (qyscore.py case)'),
        ('<solution>a,b,c,d</solution>', 'a,b,c,d', 1.0, 'Single group of 4'),
        ('\\boxed{\\text{cat, dog, bird, fish}}', 'cat,dog,bird,fish', 1.0, 'Boxed format'),
        ('no solution here', 'a,b,c,d', 0.0, 'No solution found'),
    ]

    passed = 0
    for output, gt, expected, desc in test_cases:
        result = connections_process_results(gt, output)
        status = '✓' if result == expected else '✗'
        if result == expected:
            passed += 1
        print(f'  {status} {desc}: {result} (expected {expected})', flush=True)

    print(f'\nPassed: {passed}/{len(test_cases)}', flush=True)
    return passed == len(test_cases)


def test_unscrambling():
    """Test plot unscrambling evaluator - aligned with qyscore.py test cases."""
    print('\n=== Unscrambling Evaluator Tests ===\n', flush=True)

    # Test cases from qyscore.py
    gt_plot = "The hero wakes up. He fights the dragon. He wins the gold."
    ans_plot_perfect = "<PLOT_SUMMARY>The hero wakes up. He fights the dragon. He wins the gold.</PLOT_SUMMARY>"
    ans_plot_swapped = "<PLOT_SUMMARY>The hero wakes up. He wins the gold. He fights the dragon.</PLOT_SUMMARY>"

    test_cases = [
        # (model_output, ground_truth, expected_score, description)
        (ans_plot_perfect, gt_plot, 1.0, 'Perfect order (qyscore.py case)'),
        (ans_plot_swapped, gt_plot, 0.33, 'Swapped last two sentences (qyscore.py case)'),
        ('<PLOT_SUMMARY>A. B. C.</PLOT_SUMMARY>', 'A. B. C.', 1.0, 'Short sentences'),
    ]

    passed = 0
    for output, gt, expected, desc in test_cases:
        result = plot_unscrambling_process_results(gt, output)
        # Use approximate matching for unscrambling (Levenshtein-based)
        is_close = abs(result - expected) < 0.1
        status = '✓' if is_close else '✗'
        if is_close:
            passed += 1
        print(f'  {status} {desc}: {result:.2f} (expected ~{expected})', flush=True)

    print(f'\nPassed: {passed}/{len(test_cases)}', flush=True)
    return passed == len(test_cases)


def test_via_verl_compute_score():
    """Test via verl_compute_score interface with QY data format."""
    print('\n=== verl_compute_score Interface Tests (QY Format) ===\n', flush=True)

    # QY format: data_source is task type, extra_info has "label" field
    test_cases = [
        ('typos', '<solution>hello</solution>', {'label': 'hello'}, 1.0),
        ('connections', '<solution>a,b,c,d</solution>', {'label': 'a,b,c,d'}, 1.0),
        ('unscrambling', '<PLOT_SUMMARY>A. B.</PLOT_SUMMARY>', {'label': 'A. B.'}, 1.0),
    ]

    passed = 0
    for data_source, output, extra_info, expected in test_cases:
        result = verl_compute_score(data_source, output, '', extra_info)
        status = '✓' if result == expected else '✗'
        if result == expected:
            passed += 1
        print(f'  {status} {data_source}: {result} (expected {expected})', flush=True)

    print(f'\nPassed: {passed}/{len(test_cases)}', flush=True)
    return passed == len(test_cases)


def test_threaded():
    """Test thread safety with concurrent execution."""
    print('\n=== Thread Safety Tests ===\n', flush=True)

    num_workers = 8
    num_tasks = 32

    def test_qy(i):
        # QY format: data_source is task type, extra_info has "label"
        return verl_compute_score(
            'typos',
            f'<solution>word{i}</solution>',
            '',
            {'label': f'word{i}'}
        )

    print(f'Running {num_tasks} tasks with {num_workers} workers...', flush=True)

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(test_qy, i) for i in range(num_tasks)]
        results = [f.result(timeout=60) for f in futures]
    elapsed = time.time() - start

    all_correct = all(r == 1.0 for r in results)
    print(f'  Results: {sum(r == 1.0 for r in results)}/{len(results)} correct', flush=True)
    print(f'  Time: {elapsed:.2f}s', flush=True)
    print(f'  Status: {"✓ PASSED" if all_correct else "✗ FAILED"}', flush=True)

    return all_correct


def test_edge_cases():
    """Test edge cases and error handling."""
    print('\n=== Edge Case Tests ===\n', flush=True)

    # Valid case
    print('Valid typos case:', flush=True)
    result = typos_score_fn('test', {'label': 'test'})
    print(f'  Result: {result} (expected 1.0)', flush=True)

    # Empty model output
    print('Empty model output:', flush=True)
    result = typos_score_fn('', {'label': 'test'})
    print(f'  Result: {result} (expected 0.0)', flush=True)


def main():
    print('=' * 60, flush=True)
    print('QY DOMAIN SCORE TEST SUITE', flush=True)
    print('=' * 60, flush=True)

    all_passed = True

    all_passed &= test_typos()
    all_passed &= test_connections()
    all_passed &= test_unscrambling()
    all_passed &= test_via_verl_compute_score()
    all_passed &= test_threaded()
    test_edge_cases()

    print('\n' + '=' * 60, flush=True)
    if all_passed:
        print('ALL TESTS PASSED', flush=True)
        sys.exit(0)
    else:
        print('SOME TESTS FAILED', flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
