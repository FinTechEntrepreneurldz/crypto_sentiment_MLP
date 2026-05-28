from qsentia_btc_sentiment_ensemble_ibkr.signal_engine import apply_btc_shock_override


def _neutral_components():
    return {
        "mlp_signal": 0,
        "mlp_confidence": 0.9,
        "baseline_signal": 0,
        "baseline_confidence": 0.4,
        "mean_signal": 0,
        "mean_confidence": 0.1,
    }


def test_btc_crash_override_forces_short_floor():
    signal, confidence, diag = apply_btc_shock_override(
        signal=0,
        confidence=0.02,
        components=_neutral_components(),
        shock_metrics={"btc_ret_24h": -0.032, "btc_ret_5d": -0.05, "btc_shock_source": "test"},
        text_rows=100,
        min_text_rows=5,
        enabled=True,
        confidence_floor=0.12,
    )
    assert signal == -1
    assert confidence == 0.12
    assert diag["applied"] is True
    assert diag["reason"] == "btc_crash_risk_off"


def test_btc_rally_override_forces_long_floor():
    signal, confidence, diag = apply_btc_shock_override(
        signal=0,
        confidence=0.02,
        components=_neutral_components(),
        shock_metrics={"btc_ret_24h": 0.031, "btc_ret_5d": 0.05, "btc_shock_source": "test"},
        text_rows=100,
        min_text_rows=5,
        enabled=True,
        rally_confidence_floor=0.13,
    )
    assert signal == 1
    assert confidence == 0.13
    assert diag["applied"] is True
    assert diag["reason"] == "btc_rally_risk_on"


def test_btc_crash_override_blocks_on_clear_bullish_stack():
    components = _neutral_components()
    components["mlp_signal"] = 1
    components["mlp_confidence"] = 0.95
    signal, confidence, diag = apply_btc_shock_override(
        signal=0,
        confidence=0.02,
        components=components,
        shock_metrics={"btc_ret_24h": -0.04, "btc_ret_5d": -0.06},
        text_rows=100,
        min_text_rows=5,
        enabled=True,
        confidence_floor=0.12,
    )
    assert signal == 0
    assert confidence == 0.02
    assert diag["applied"] is False
    assert diag["reason"] == "blocked_by_opposed_components"


def test_btc_rally_override_blocks_on_clear_bearish_stack():
    components = _neutral_components()
    components["mlp_signal"] = -1
    components["mlp_confidence"] = 0.95
    signal, confidence, diag = apply_btc_shock_override(
        signal=0,
        confidence=0.02,
        components=components,
        shock_metrics={"btc_ret_24h": 0.04, "btc_ret_5d": 0.06},
        text_rows=100,
        min_text_rows=5,
        enabled=True,
        rally_confidence_floor=0.12,
    )
    assert signal == 0
    assert confidence == 0.02
    assert diag["applied"] is False
    assert diag["reason"] == "blocked_by_opposed_components"
