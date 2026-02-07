"""Simple circuit breaker for exchange/API stability protection.

States:
  CLOSED  - Normal operation.
  OPEN    - Short-circuits calls; opened after threshold failures.
  HALF    - Trial state after cooldown; first success closes, failure re-opens.

Environment variables:
  CB_FAIL_THRESHOLD=5     consecutive failures to OPEN
  CB_COOLDOWN_SECONDS=30  seconds to wait before HALF-OPEN probe
  CB_HALF_MAX_ATTEMPTS=1  allowed attempts in HALF state before re-open
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class CircuitState:
	state: str = "CLOSED"  # CLOSED, OPEN, HALF
	failure_count: int = 0
	opened_at: float = 0.0
	last_failure: float = 0.0
	half_attempts: int = 0


class CircuitBreaker:
	def __init__(self, name: str,
				 fail_threshold: int = None,
				 cooldown_seconds: int = None,
				 half_max_attempts: int = None):
		import os
		self.name = name
		self.fail_threshold = fail_threshold or int(os.getenv('CB_FAIL_THRESHOLD', '5'))
		self.cooldown_seconds = cooldown_seconds or int(os.getenv('CB_COOLDOWN_SECONDS', '30'))
		self.half_max_attempts = half_max_attempts or int(os.getenv('CB_HALF_MAX_ATTEMPTS', '1'))
		self._state = CircuitState()

	# ------------- Query helpers -------------
	@property
	def state(self) -> str:
		return self._state.state

	def can_execute(self) -> bool:
		if self._state.state == 'CLOSED':
			return True
		if self._state.state == 'OPEN':
			# check cooldown expiry
			if time.time() - self._state.opened_at >= self.cooldown_seconds:
				# transition to HALF
				self._state.state = 'HALF'
				self._state.half_attempts = 0
				return True
			return False
		if self._state.state == 'HALF':
			if self._state.half_attempts < self.half_max_attempts:
				return True
			# exceeded half attempts; re-open
			self._open()
			return False
		return False

	# ------------- State transitions -------------
	def record_success(self):
		if self._state.state in ('HALF', 'OPEN'):
			self._close_reset()
			return
		# In CLOSED just reset failures
		self._state.failure_count = 0

	def record_failure(self):
		now = time.time()
		self._state.last_failure = now
		if self._state.state == 'HALF':
			self._open()
			return
		self._state.failure_count += 1
		if self._state.failure_count >= self.fail_threshold:
			self._open()

	def _open(self):
		self._state.state = 'OPEN'
		self._state.opened_at = time.time()

	def _close_reset(self):
		self._state = CircuitState(state='CLOSED')

	# ------------- Execution wrapper -------------
	def run(self, func, *args, **kwargs):
		if not self.can_execute():
			raise RuntimeError(f"Circuit '{self.name}' OPEN: rejecting operation")
		if self._state.state == 'HALF':
			self._state.half_attempts += 1
		try:
			result = func(*args, **kwargs)
			self.record_success()
			return result
		except Exception:
			self.record_failure()
			raise

	# ------------- Diagnostics -------------
	def snapshot(self) -> dict:
		return {
			'name': self.name,
			'state': self._state.state,
			'failures': self._state.failure_count,
			'opened_at': self._state.opened_at,
			'last_failure': self._state.last_failure,
			'half_attempts': self._state.half_attempts,
			'fail_threshold': self.fail_threshold,
			'cooldown_seconds': self.cooldown_seconds
		}

__all__ = ["CircuitBreaker", "CircuitState"]
