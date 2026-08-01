"""The review player's data contract.

The panel is a verification tool, so the thing that matters is that what it
displays is the same thing the measures were computed from -- and that a
missing video degrades to an explanation rather than a broken player.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from convlab.config import Config
from convlab.context import AnalysisContext
from convlab.report.player import STATE_HZ, build_player_data, render_player
from convlab.speech.attribution import AttributionResult, Calibration
from convlab.timeline import Segments
from convlab.turns import Turn, TurnSet


@pytest.fixture
def context():
    ctx = AnalysisContext("s1", Config(), 60.0, 100.0)
    n = 6001
    state = np.zeros(n, dtype=np.int8)
    state[1000:2000] = 1
    state[3000:4000] = 2
    ctx.attribution = AttributionResult(
        state=state, posterior=np.zeros((n, 4), np.float32),
        confidence=np.full(n, 0.9, np.float32), frame_hz=100.0,
        calibration=Calibration(0.0, 10.0, "gmm", ok=True),
        speech={"A": Segments.from_pairs([(10.0, 20.0)]),
                "B": Segments.from_pairs([(30.0, 40.0)])},
    )
    ctx.turn_set = TurnSet(
        turns=[Turn(index=0, person="A", start=10.0, end=20.0, text="hello there"),
               Turn(index=1, person="B", start=30.0, end=40.0, text="hi")],
        ipus=[], backchannels=[], interruptions=[], duration=60.0,
        speech=ctx.attribution.speech,
    )
    return ctx


def _videos(tmp_path):
    out = {}
    for role in ("close_a", "close_b"):
        p = tmp_path / f"{role}.mp4"
        p.write_bytes(b"\x00")
        out[role] = p
    return out


class TestPlayerData:
    def test_state_is_downsampled_to_the_declared_rate(self, context):
        data = build_player_data(context, None)
        expected = int(np.ceil(context.attribution.state.size / (100.0 / STATE_HZ)))
        assert abs(len(data["state"]) - expected) <= 1

    def test_state_preserves_who_was_speaking(self, context):
        data = build_player_data(context, None)
        # 15 s is inside A's stretch, 35 s inside B's.
        assert data["state"][int(15 * STATE_HZ)] == 1
        assert data["state"][int(35 * STATE_HZ)] == 2

    def test_turn_text_is_carried_for_the_readout(self, context):
        data = build_player_data(context, None)
        assert data["turns"][0]["x"] == "hello there"
        assert data["turns"][0]["p"] == "A"

    def test_sources_are_file_uris(self, context, tmp_path):
        data = build_player_data(context, _videos(tmp_path))
        assert data["sources"]["A"]["src"].startswith("file:///")
        assert data["sources"]["A"]["name"] == "close_a.mp4"

    def test_offset_maps_session_time_to_file_time(self, context, tmp_path):
        # close_b needs -1.7 s added to reach session time, so the file is
        # 1.7 s ahead of the session clock.
        data = build_player_data(context, _videos(tmp_path), {"close_b": -1.7})
        assert data["sources"]["B"]["offset"] == pytest.approx(1.7)
        assert data["sources"]["A"]["offset"] == pytest.approx(0.0)

    def test_payload_is_json_serialisable(self, context, tmp_path):
        data = build_player_data(context, _videos(tmp_path))
        json.dumps(data)  # must not raise on numpy scalars

    def test_no_attribution_yields_an_empty_track(self):
        ctx = AnalysisContext("s", Config(), 10.0, 100.0)
        assert build_player_data(ctx, None)["state"] == []


class TestPlayerHtml:
    def test_missing_video_explains_itself(self, context):
        html = render_player(build_player_data(context, None))
        assert "<video" not in html
        assert "not available" in html

    def test_video_present_produces_two_players(self, context, tmp_path):
        html = render_player(build_player_data(context, _videos(tmp_path)))
        assert html.count("<video") == 2
        assert 'id="vidA"' in html and 'id="vidB"' in html

    @staticmethod
    def _embedded(html: str) -> dict:
        """Pull the payload back out exactly as an HTML parser would."""
        opener = 'type="application/json">'
        start = html.index(opener) + len(opener)
        end = html.index("</script>", start)  # first one wins, as in a browser
        return json.loads(html[start:end].replace("<\\/", "</"))

    def test_embedded_json_parses(self, context, tmp_path):
        html = render_player(build_player_data(context, _videos(tmp_path)))
        assert self._embedded(html)["duration"] == 60.0

    def test_a_transcript_containing_a_script_tag_cannot_truncate_the_page(
        self, context, tmp_path
    ):
        """Recogniser output is untrusted text and can contain anything.

        An HTML parser ends the script block at the first literal
        ``</script>``, whatever the JSON quoting says, so an unescaped one
        would cut the payload in half and break the report.
        """
        context.turn_set.turns[0] = Turn(
            index=0, person="A", start=10.0, end=20.0,
            text="then he said </script><img src=x onerror=alert(1)>",
        )
        html = render_player(build_player_data(context, _videos(tmp_path)))
        payload = self._embedded(html)
        assert payload["duration"] == 60.0, "payload must survive intact"
        assert "</script>" in payload["turns"][0]["x"], "text must be preserved"

    def test_ordinary_text_is_unchanged(self, context, tmp_path):
        html = render_player(build_player_data(context, _videos(tmp_path)))
        assert self._embedded(html)["turns"][0]["x"] == "hello there"
