from qsentia_btc_sentiment_ensemble_ibkr.risk import target_mbt_contracts


def test_target_contracts_long():
    order = target_mbt_contracts(
        signal=1,
        confidence=0.5,
        net_liq=100_000,
        btc_price=100_000,
        target_gross_fraction=0.2,
        max_contracts=10,
        allow_short=True,
        min_confidence=0.1,
    )
    assert order.desired_contracts == 1
    assert order.action == "buy"


def test_blocks_short_when_disabled():
    order = target_mbt_contracts(
        signal=-1,
        confidence=0.8,
        net_liq=100_000,
        btc_price=100_000,
        target_gross_fraction=0.5,
        max_contracts=10,
        allow_short=False,
        min_confidence=0.1,
    )
    assert order.desired_contracts == 0
    assert order.reason == "short_signal_blocked"

