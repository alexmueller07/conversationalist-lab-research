"""Memory probing and recognizer sizing.

These exist because the failure they prevent is silent and expensive: the
process is killed part-way through a batch, with no traceback, after an hour
of tracking work has already been done.
"""

from __future__ import annotations

import pytest

from convlab.system import (
    ASR_MODEL_COMMIT_MB,
    SAFETY_MARGIN_MB,
    available_memory_mb,
    fit_asr_model,
)


class TestAvailableMemory:
    def test_returns_a_plausible_number_or_none(self):
        value = available_memory_mb()
        assert value is None or (0 < value < 4_000_000)

    def test_never_raises(self, monkeypatch):
        monkeypatch.setattr("platform.system", lambda: "Plan9")
        assert available_memory_mb() is None


class TestFitAsrModel:
    def test_keeps_the_request_when_memory_is_plentiful(self):
        model, note = fit_asr_model("small.en", available_mb=16_000)
        assert model == "small.en" and note == ""

    def test_steps_down_when_memory_is_short(self):
        # Enough for base.en (1050 + margin) but not small.en (2400 + margin).
        available = ASR_MODEL_COMMIT_MB["base.en"] + SAFETY_MARGIN_MB + 50
        model, note = fit_asr_model("small.en", available_mb=available)
        assert model == "base.en"
        assert "not enough" in note and "small.en" in note

    def test_steps_down_twice_when_very_short(self):
        available = ASR_MODEL_COMMIT_MB["tiny.en"] + SAFETY_MARGIN_MB + 10
        model, note = fit_asr_model("small.en", available_mb=available)
        assert model == "tiny.en"

    def test_warns_when_even_the_smallest_will_not_fit(self):
        model, note = fit_asr_model("small.en", available_mb=100)
        assert model == "tiny.en"
        assert "may fail" in note

    def test_boundary_is_inclusive_of_the_margin(self):
        exact = ASR_MODEL_COMMIT_MB["small.en"] + SAFETY_MARGIN_MB
        assert fit_asr_model("small.en", available_mb=exact)[0] == "small.en"
        assert fit_asr_model("small.en", available_mb=exact - 1)[0] != "small.en"

    def test_unknown_model_is_left_alone(self):
        model, note = fit_asr_model("large-v3", available_mb=100)
        assert model == "large-v3" and note == ""

    def test_unknown_memory_leaves_the_request_alone(self, monkeypatch):
        monkeypatch.setattr("convlab.system.available_memory_mb", lambda: None)
        assert fit_asr_model("small.en")[0] == "small.en"

    def test_multilingual_request_stays_multilingual(self):
        model, _ = fit_asr_model("small", available_mb=1800)
        assert not model.endswith(".en"), "must not silently switch language mode"

    @pytest.mark.parametrize("name", sorted(ASR_MODEL_COMMIT_MB))
    def test_every_known_model_fits_with_enough_memory(self, name):
        assert fit_asr_model(name, available_mb=64_000)[0] == name
