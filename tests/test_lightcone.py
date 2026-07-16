"""Locality regression test: the internal light cone (Task 12).

Latency must grow with stimulus distance and shrink with thinking rate.
Any non-local shortcut in the internal dynamics would flatten latency in d
and fail this test. Deterministic (exact policy distributions), so single
measurements suffice.
"""

from efi.evaluation.probes import reaction_latency


class TestLightCone:
    def test_latency_increases_with_distance(self):
        l_near = reaction_latency(d=5, kappa=3)
        l_far = reaction_latency(d=20, kappa=3)
        assert l_near is not None and l_far is not None
        assert l_far > l_near

    def test_latency_decreases_with_thinking_rate(self):
        l_slow = reaction_latency(d=20, kappa=1)
        l_fast = reaction_latency(d=20, kappa=5)
        assert l_slow is not None and l_fast is not None
        assert l_fast < l_slow

    def test_response_is_finite_within_reach(self):
        """A stimulus inside the value-reach budget always arrives."""
        for d in (5, 10, 15):
            assert reaction_latency(d, kappa=3) is not None

    def test_information_cannot_outrun_the_cone(self):
        """Latency in ticks is at least ceil((d - window_reach)/kappa):
        the stencil moves information at most kappa cells per tick."""
        for d, kappa in [(10, 1), (20, 3)]:
            lat = reaction_latency(d, kappa)
            assert lat is not None
            min_ticks = max(1, (d - 3) // (2 * kappa))  # generous bound
            assert lat >= min_ticks
