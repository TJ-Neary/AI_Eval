"""
Async State Machine

Generic finite state machine with transition history and async callbacks.
Useful for managing lifecycle states in interactive systems (voice pipelines,
CLI workflows, game loops, process orchestration).

Usage:
    from enum import Enum, auto

    class AppState(Enum):
        IDLE = auto()
        PROCESSING = auto()
        ERROR = auto()

    sm = StateMachine(initial_state=AppState.IDLE)

    # Optional: register a callback
    async def on_change(old, new, reason):
        print(f"{old.name} -> {new.name}: {reason}")
    sm.on_transition(on_change)

    # Transition
    await sm.transition_to(AppState.PROCESSING, reason="user request")
    print(sm.state)  # AppState.PROCESSING
    print(sm.history)  # list of StateTransition records

Contributed by: Kendra
"""

import logging
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Awaitable, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)

S = TypeVar("S", bound=Enum)


@dataclass
class StateTransition(Generic[S]):
    """Records a single state transition."""
    from_state: S
    to_state: S
    timestamp: datetime
    reason: str


class StateMachine(Generic[S]):
    """
    Generic async state machine with transition history.

    Features:
    - Enum-based states (any Enum subclass)
    - Duplicate transitions are no-ops
    - Optional allowed_transitions map for enforcement
    - Async callback on every transition
    - Rolling history (configurable max size)
    """

    def __init__(
        self,
        initial_state: S,
        allowed_transitions: Optional[dict[S, List[S]]] = None,
        max_history: int = 100,
    ):
        """
        Args:
            initial_state: The starting state.
            allowed_transitions: Optional dict mapping each state to its valid
                                 target states. If None, all transitions allowed.
            max_history: Max number of transitions to keep in history.
        """
        self._state: S = initial_state
        self._allowed = allowed_transitions
        self._max_history = max_history
        self._history: List[StateTransition[S]] = []
        self._callback: Optional[Callable[[S, S, str], Awaitable[None]]] = None

    @property
    def state(self) -> S:
        """Current state."""
        return self._state

    @property
    def history(self) -> List[StateTransition[S]]:
        """Transition history (read-only copy)."""
        return list(self._history)

    def on_transition(self, callback: Callable[[S, S, str], Awaitable[None]]) -> None:
        """Register an async callback: (old_state, new_state, reason) -> None."""
        self._callback = callback

    async def transition_to(self, new_state: S, reason: str = "") -> bool:
        """
        Transition to a new state.

        Args:
            new_state: Target state.
            reason: Human-readable reason (for debugging/logging).

        Returns:
            True if the transition occurred, False if skipped (duplicate or invalid).
        """
        if new_state == self._state:
            return False

        # Enforce allowed transitions if configured
        if self._allowed is not None:
            allowed = self._allowed.get(self._state, [])
            if new_state not in allowed:
                logger.warning(
                    f"Invalid transition: {self._state.name} -> {new_state.name} "
                    f"(allowed: {[s.name for s in allowed]})"
                )
                return False

        old_state = self._state
        self._state = new_state

        transition = StateTransition(
            from_state=old_state,
            to_state=new_state,
            timestamp=datetime.now(),
            reason=reason,
        )
        self._history.append(transition)

        # Trim history
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        logger.info(f"State: {old_state.name} -> {new_state.name} ({reason})")

        if self._callback:
            await self._callback(old_state, new_state, reason)

        return True

    def get_status(self) -> dict:
        """Get current state and recent transitions as a dict."""
        return {
            "state": self._state.name,
            "last_transitions": [
                {
                    "from": t.from_state.name,
                    "to": t.to_state.name,
                    "timestamp": t.timestamp.isoformat(),
                    "reason": t.reason,
                }
                for t in self._history[-5:]
            ],
        }
