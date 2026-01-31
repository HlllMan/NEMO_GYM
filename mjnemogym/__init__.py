"""
MJ NeMo Gym - Scoring functions for various tasks
"""

from mjnemogym.math_with_judge.app_copy import score_fn as math_score_fn
from mjnemogym.code_gen.app_copy import score_fn as code_score_fn
from mjnemogym.mcqa.app_copy import score_fn as mcqa_score_fn
from mjnemogym.instruction_following.app_copy import score_fn as if_score_fn
from mjnemogym.structured_outputs.app_copy import score_fn as so_score_fn
from mjnemogym.workspace_assistant.app_copy import score_fn as wa_score_fn

score_fn_dict = {
    "math": math_score_fn,
    "code": code_score_fn,
    "mcqa": mcqa_score_fn,
    "if": if_score_fn,
    "so": so_score_fn,
    "wa": wa_score_fn,
}

__all__ = [
    "math_score_fn",
    "code_score_fn",
    "mcqa_score_fn",
    "if_score_fn",
    "so_score_fn",
    "wa_score_fn",
    "score_fn_dict",
]
