"""Contract formulation: natural-language requirements to ALTLf formulas.

The pipeline follows a lift-then-ground decomposition. A language model
first lifts each requirement to a structured intermediate representation
(:mod:`contragent.generation.structured_ir`) whose leaves are
placeholders; a deterministic step then grounds the placeholders to
interaction predicates and produces the closed formula. Each formula is
translated back to natural language for review
(:mod:`contragent.formulas.nl_gen`).
"""

from contragent.generation.llm_extraction import (
    ExtractionResult,
    UnifiedExtractor,
    compile_extraction,
    register_atom,
    register_atoms,
)
from contragent.generation.structured_ir import (
    ConstraintIR,
    IRCompilationResult,
    compile_ir,
    compile_ir_batch,
)

__all__ = [
    "ExtractionResult",
    "UnifiedExtractor",
    "compile_extraction",
    "register_atom",
    "register_atoms",
    "ConstraintIR",
    "IRCompilationResult",
    "compile_ir",
    "compile_ir_batch",
]
