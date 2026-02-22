"""Tests for declick module."""

from pathlib import Path

import numpy as np
import pytest

from audio_tools.declick import detect_clicks, repair_clicks, process_audio


class TestDetectClicks:
    def test_no_clicks_in_smooth_signal(self):
        # Smooth ramp - no clicks
        samples = np.linspace(0, 1, 100)
        clicks = detect_clicks(samples)
        assert len(clicks) == 0

    def test_no_clicks_in_sine_wave(self):
        # Sine wave - no clicks expected
        t = np.linspace(0, 2 * np.pi, 1000)
        samples = np.sin(t)
        clicks = detect_clicks(samples)
        assert len(clicks) == 0

    def test_detects_obvious_click(self):
        # Create a smooth signal with one obvious click
        samples = np.zeros(100)
        samples[50] = 1.0  # Huge spike
        clicks = detect_clicks(samples, ratio_threshold=5.0, min_diff_threshold=0.001)
        assert 50 in clicks

    def test_detects_click_in_smooth_ramp(self):
        # Smooth ramp with a click in the middle
        samples = np.linspace(0, 0.5, 100)
        samples[50] = 0.9  # Click - way off from expected ~0.25
        clicks = detect_clicks(samples, ratio_threshold=5.0, min_diff_threshold=0.01)
        assert 50 in clicks

    def test_ignores_small_noise(self):
        # Small variations below min_diff should be ignored
        samples = np.zeros(100)
        samples[50] = 0.005  # Below default threshold
        clicks = detect_clicks(samples, min_diff_threshold=0.01)
        assert len(clicks) == 0

    def test_empty_or_short_array(self):
        assert len(detect_clicks(np.array([]))) == 0
        assert len(detect_clicks(np.array([1.0]))) == 0
        assert len(detect_clicks(np.array([1.0, 2.0]))) == 0


class TestRepairClicks:
    def test_repairs_click_with_interpolation(self):
        samples = np.array([0.0, 0.1, 0.9, 0.3, 0.4])  # Click at index 2
        clicks = np.array([2])
        repaired = repair_clicks(samples, clicks)
        # Should interpolate: (0.1 + 0.3) / 2 = 0.2
        assert repaired[2] == pytest.approx(0.2)
        # Other samples unchanged
        assert repaired[0] == 0.0
        assert repaired[1] == 0.1
        assert repaired[3] == 0.3
        assert repaired[4] == 0.4

    def test_repairs_multiple_clicks(self):
        samples = np.array([0.0, 0.1, 0.9, 0.3, 0.8, 0.5])  # Clicks at 2 and 4
        clicks = np.array([2, 4])
        repaired = repair_clicks(samples, clicks)
        assert repaired[2] == pytest.approx(0.2)  # (0.1 + 0.3) / 2
        assert repaired[4] == pytest.approx(0.4)  # (0.3 + 0.5) / 2

    def test_no_clicks_returns_copy(self):
        samples = np.array([1.0, 2.0, 3.0])
        repaired = repair_clicks(samples, np.array([]))
        np.testing.assert_array_equal(samples, repaired)
        # Ensure it's a copy, not the same array
        repaired[0] = 999
        assert samples[0] == 1.0

    def test_doesnt_modify_original(self):
        samples = np.array([0.0, 0.1, 0.9, 0.3, 0.4])
        original = samples.copy()
        clicks = np.array([2])
        repair_clicks(samples, clicks)
        np.testing.assert_array_equal(samples, original)


class TestProcessAudio:
    def test_mono_processing(self):
        # Mono signal with a click
        data = np.zeros(100)
        data[50] = 1.0
        repaired, count = process_audio(data, ratio_threshold=5.0, min_diff_threshold=0.001)
        assert count >= 1
        assert repaired[50] == pytest.approx(0.0)

    def test_stereo_processing(self):
        # Stereo signal with clicks in different channels
        data = np.zeros((100, 2))
        data[30, 0] = 1.0  # Click in left channel
        data[60, 1] = 1.0  # Click in right channel
        repaired, count = process_audio(data, ratio_threshold=5.0, min_diff_threshold=0.001)
        assert count >= 2
        assert repaired[30, 0] == pytest.approx(0.0)
        assert repaired[60, 1] == pytest.approx(0.0)

    def test_no_clicks_in_clean_audio(self):
        # Clean sine wave
        t = np.linspace(0, 4 * np.pi, 1000)
        data = np.sin(t)
        repaired, count = process_audio(data)
        assert count == 0
        np.testing.assert_array_almost_equal(data, repaired)


class TestDropoutDetection:
    """Tests for digital dropout detection (zeros in non-zero signal)."""

    def test_detects_single_dropout(self):
        # Smooth signal with one zero dropout
        samples = np.full(100, 0.5)
        samples[50] = 0.0  # Dropout
        clicks = detect_clicks(samples)
        assert 50 in clicks

    def test_detects_consecutive_dropouts(self):
        # Smooth signal with multiple consecutive zeros (like ADAT dropout)
        samples = np.full(100, 0.5)
        samples[50] = 0.0
        samples[51] = 0.0
        samples[52] = 0.0
        clicks = detect_clicks(samples)
        assert 50 in clicks
        assert 51 in clicks
        assert 52 in clicks

    def test_ignores_natural_zero_crossings(self):
        # Sine wave naturally crosses zero - shouldn't flag those
        t = np.linspace(0, 4 * np.pi, 1000)
        samples = 0.5 * np.sin(t)
        clicks = detect_clicks(samples)
        assert len(clicks) == 0

    def test_dropout_in_varying_signal(self):
        # Dropout in a signal that's not constant
        t = np.linspace(0, 2 * np.pi, 500)
        samples = 0.3 * np.sin(t) + 0.4  # Oscillates 0.1 to 0.7
        samples[250] = 0.0  # Dropout when signal is around 0.4
        clicks = detect_clicks(samples)
        assert 250 in clicks


class TestNearZeroDropouts:
    """Tests for dropouts that aren't exactly zero - the bug we missed."""

    def test_detects_near_zero_with_dc_offset(self):
        """Dropout with small DC offset should still be caught."""
        samples = np.full(100, 0.5)
        # Dropout with 1mV DC offset - not exactly zero
        samples[50] = 0.001
        clicks = detect_clicks(samples)
        assert 50 in clicks, "Failed to detect near-zero dropout with DC offset"

    def test_detects_near_zero_with_dither(self):
        """Dropout with dither-level noise should still be caught."""
        samples = np.full(100, 0.4)
        # Dropout at dither noise floor level (~-80dB)
        samples[50] = 0.0001
        clicks = detect_clicks(samples)
        assert 50 in clicks, "Failed to detect dropout at dither noise level"

    def test_detects_dip_pattern(self):
        """Sample that's much smaller than neighbors (partial dropout)."""
        samples = np.full(100, 0.3)
        # Partial dropout - reduced to 10% of signal but not zero
        samples[50] = 0.02
        clicks = detect_clicks(samples)
        assert 50 in clicks, "Failed to detect partial dropout (dip pattern)"

    def test_detects_dip_in_varying_signal(self):
        """Dip detection should work in non-constant signals."""
        t = np.linspace(0, 2 * np.pi, 500)
        samples = 0.3 * np.sin(t) + 0.4  # Oscillates 0.1 to 0.7
        # At sample 125, signal is ~0.6. Make it drop to 0.03
        samples[125] = 0.03
        clicks = detect_clicks(samples)
        assert 125 in clicks, "Failed to detect dip in varying signal"

    def test_no_false_positive_at_zero_crossing(self):
        """Don't flag natural zero crossings as dips."""
        t = np.linspace(0, 4 * np.pi, 1000)
        samples = 0.5 * np.sin(t)
        clicks = detect_clicks(samples)
        assert len(clicks) == 0, "False positive at natural zero crossing"

    def test_no_false_positive_in_quiet_section(self):
        """Don't flag legitimately quiet sections."""
        samples = np.zeros(100)
        # Quiet section with small signal
        samples[40:60] = 0.001
        samples[50] = 0.0001  # Slightly quieter in the middle - NOT a dropout
        clicks = detect_clicks(samples)
        assert 50 not in clicks, "False positive in legitimately quiet signal"

    def test_consecutive_near_zero_dropouts(self):
        """Multiple consecutive near-zero samples should all be caught."""
        samples = np.full(100, 0.5)
        samples[50] = 0.002
        samples[51] = 0.001
        samples[52] = 0.003
        clicks = detect_clicks(samples)
        assert 50 in clicks
        assert 51 in clicks
        assert 52 in clicks

    def test_mixed_zero_and_near_zero(self):
        """Multi-sample dropout with mix of exact and near zeros."""
        samples = np.full(100, 0.5)
        samples[50] = 0.0    # Exact zero
        samples[51] = 0.002  # Near zero
        samples[52] = 0.0    # Exact zero
        clicks = detect_clicks(samples)
        assert 50 in clicks
        assert 51 in clicks, "Failed to detect near-zero between exact zeros"
        assert 52 in clicks


class TestConsecutiveClickRepair:
    """Tests for repairing consecutive clicks/dropouts."""

    def test_repairs_consecutive_clicks_with_interpolation(self):
        # Signal with 3 consecutive bad samples
        samples = np.array([0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.6, 0.7, 0.8])
        clicks = np.array([3, 4, 5])
        repaired = repair_clicks(samples, clicks)
        # Should interpolate linearly from 0.2 to 0.6
        assert repaired[3] == pytest.approx(0.3)
        assert repaired[4] == pytest.approx(0.4)
        assert repaired[5] == pytest.approx(0.5)

    def test_repairs_two_consecutive_dropouts(self):
        # Common case: two consecutive zeros
        samples = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5])
        clicks = np.array([3, 4])
        repaired = repair_clicks(samples, clicks)
        # Linear interpolation from index 2 (0.5) to index 5 (0.5)
        assert repaired[3] == pytest.approx(0.5)
        assert repaired[4] == pytest.approx(0.5)

    def test_non_consecutive_clicks_repaired_independently(self):
        # Two separate clicks should be repaired from their own neighbors
        samples = np.array([0.1, 0.2, 0.9, 0.4, 0.5, 0.6, 0.0, 0.8, 0.9])
        clicks = np.array([2, 6])
        repaired = repair_clicks(samples, clicks)
        assert repaired[2] == pytest.approx(0.3)  # (0.2 + 0.4) / 2
        assert repaired[6] == pytest.approx(0.7)  # (0.6 + 0.8) / 2


class TestSyntheticIntegration:
    def test_spike_clicks(self):
        # Smooth audio with injected spike clicks
        t = np.linspace(0, 2 * np.pi, 1000)
        clean = 0.5 * np.sin(t) + 0.1 * np.sin(3 * t)

        corrupted = clean.copy()
        click_positions = [100, 300, 500, 700]
        for pos in click_positions:
            corrupted[pos] = np.sign(clean[pos]) * 0.95

        repaired, count = process_audio(corrupted, ratio_threshold=8.0, min_diff_threshold=0.01)

        assert count >= len(click_positions) - 1
        for pos in click_positions:
            assert abs(repaired[pos] - clean[pos]) < abs(corrupted[pos] - clean[pos])

    def test_dropout_clicks(self):
        # Smooth audio with injected zero dropouts (ADAT-style)
        t = np.linspace(0, 4 * np.pi, 2000)
        clean = 0.4 * np.sin(t) + 0.2 * np.sin(2.5 * t)

        corrupted = clean.copy()
        # Single dropout
        corrupted[400] = 0.0
        # Double dropout
        corrupted[800] = 0.0
        corrupted[801] = 0.0
        # Triple dropout
        corrupted[1200] = 0.0
        corrupted[1201] = 0.0
        corrupted[1202] = 0.0

        repaired, count = process_audio(corrupted)

        assert count >= 6  # Should find all dropouts
        # Repaired values should be closer to clean
        for pos in [400, 800, 801, 1200, 1201, 1202]:
            assert abs(repaired[pos] - clean[pos]) < abs(corrupted[pos] - clean[pos])

    def test_near_zero_dropouts_with_dc_offset(self):
        """Regression test: dropouts that aren't exactly zero due to DC offset."""
        t = np.linspace(0, 4 * np.pi, 2000)
        clean = 0.4 * np.sin(t) + 0.2 * np.sin(2.5 * t)

        corrupted = clean.copy()
        dc_offset = 0.002  # Small DC offset makes dropouts non-zero

        # Single near-zero dropout
        corrupted[400] = dc_offset
        # Double near-zero dropout
        corrupted[800] = dc_offset
        corrupted[801] = dc_offset * 1.5
        # Triple near-zero dropout
        corrupted[1200] = dc_offset
        corrupted[1201] = dc_offset * 0.5
        corrupted[1202] = dc_offset * 2

        repaired, count = process_audio(corrupted)

        assert count >= 6, f"Only detected {count} of 6 near-zero dropouts"
        for pos in [400, 800, 801, 1200, 1201, 1202]:
            assert abs(repaired[pos] - clean[pos]) < abs(corrupted[pos] - clean[pos]), \
                f"Failed to improve sample at {pos}"

    def test_partial_dropouts_dip_pattern(self):
        """Regression test: partial dropouts where signal dips but doesn't reach zero."""
        t = np.linspace(0, 4 * np.pi, 2000)
        clean = 0.4 * np.sin(t) + 0.35  # Oscillates between ~0 and ~0.75

        corrupted = clean.copy()
        # Partial dropout - signal drops to ~5% of expected value
        # At pos 500, clean signal is around 0.55
        corrupted[500] = 0.02
        # At pos 1000, clean signal is around 0.35
        corrupted[1000] = 0.015
        # At pos 1500, clean signal is around 0.55
        corrupted[1500] = 0.025

        repaired, count = process_audio(corrupted)

        assert count >= 3, f"Only detected {count} of 3 partial dropouts"
        for pos in [500, 1000, 1500]:
            assert abs(repaired[pos] - clean[pos]) < abs(corrupted[pos] - clean[pos]), \
                f"Failed to improve partial dropout at {pos}"

    def test_realistic_adat_dropout_pattern(self):
        """Simulate realistic ADAT sync error: 1-8 sample dropout with edge artifacts."""
        np.random.seed(42)
        t = np.linspace(0, 8 * np.pi, 4000)
        clean = 0.3 * np.sin(t) + 0.15 * np.sin(2.7 * t) + 0.25

        corrupted = clean.copy()
        dropout_positions = []

        # ADAT dropout 1: 3 samples, exact zeros
        for i in range(3):
            corrupted[1000 + i] = 0.0
            dropout_positions.append(1000 + i)

        # ADAT dropout 2: 5 samples, near-zeros with slight variation
        for i in range(5):
            corrupted[2000 + i] = np.random.uniform(0, 0.003)
            dropout_positions.append(2000 + i)

        # ADAT dropout 3: 2 samples, one exact zero, one near-zero
        corrupted[3000] = 0.0
        corrupted[3001] = 0.001
        dropout_positions.extend([3000, 3001])

        repaired, count = process_audio(corrupted)

        assert count >= len(dropout_positions) - 1, \
            f"Detected {count} of {len(dropout_positions)} dropout samples"

        for pos in dropout_positions:
            orig_error = abs(corrupted[pos] - clean[pos])
            fixed_error = abs(repaired[pos] - clean[pos])
            assert fixed_error < orig_error, \
                f"Repair at {pos} made things worse: {orig_error:.4f} -> {fixed_error:.4f}"


class TestRealAudio:
    """Integration tests using real audio file - just verifies basic functionality."""

    FIXTURE_PATH = Path(__file__).parent / "fixtures" / "click_sample.wav"

    @pytest.fixture
    def audio_data(self):
        import soundfile as sf
        if not self.FIXTURE_PATH.exists():
            pytest.skip("Fixture file not found")
        data, sr = sf.read(self.FIXTURE_PATH)
        return data, sr

    def test_detects_and_repairs_clicks(self, audio_data):
        data, sr = audio_data
        clicks = detect_clicks(data)
        repaired = repair_clicks(data, clicks)

        # Should find some clicks
        assert len(clicks) >= 1

        # Repaired samples should differ from original
        for idx in clicks:
            assert repaired[idx] != data[idx]

        # Non-click samples unchanged
        click_set = set(clicks)
        for i in [100, 1000, 10000, 50000]:
            if i not in click_set:
                assert repaired[i] == data[i]

    def test_cli_processes_file(self, audio_data, tmp_path):
        import subprocess
        import sys
        output_path = tmp_path / "output.wav"
        result = subprocess.run(
            [sys.executable, "-m", "audio_tools.declick", "-d", str(self.FIXTURE_PATH), "-o", str(output_path), "-v"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert output_path.exists()
