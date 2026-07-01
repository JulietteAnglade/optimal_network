"""Top-level orchestrator (Section 5.1, step 1 + step 2 loop)."""
from .algo_params import AlgoParams
from .outer_loop import OuterLoopBlock
from .params import ModelParams
from .state import ModelState


class GeneralizedSpatialEquilibriumSolver:
    """Initializes the outer-loop state and runs the outer loop to
    convergence (or until AlgoParams.T_outer cycles have elapsed)."""

    def __init__(self):
        self.outer_block = OuterLoopBlock()

    @property
    def history(self):
        return self.outer_block.history

    def solve(self, initial_state: ModelState, params: ModelParams, algo: AlgoParams) -> ModelState:
        state = initial_state
        state = self.outer_block.solve(state, params, algo)
        return state
