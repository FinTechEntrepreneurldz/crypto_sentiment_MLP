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


def test_low_confidence_holds_existing_position_by_default():
    order = target_mbt_contracts(
        signal=-1,
        confidence=0.099,
        net_liq=1_000_000,
        btc_price=75_000,
        target_gross_fraction=0.9,
        max_contracts=150,
        allow_short=True,
        min_confidence=0.1,
        current_contracts=-65,
    )
    assert order.desired_contracts == -65
    assert order.delta_contracts == 0
    assert order.action == "hold"
    assert order.reason == "hold_existing_low_confidence"


def test_low_confidence_can_flatten_when_explicitly_enabled():
    order = target_mbt_contracts(
        signal=-1,
        confidence=0.099,
        net_liq=1_000_000,
        btc_price=75_000,
        target_gross_fraction=0.9,
        max_contracts=150,
        allow_short=True,
        min_confidence=0.1,
        current_contracts=-65,
        flatten_on_low_confidence=True,
    )
    assert order.desired_contracts == 0
    assert order.delta_contracts == 65
    assert order.action == "buy"
    assert order.reason == "flat_or_low_confidence"


def test_confirmed_signal_flips_in_one_delta_order():
    order = target_mbt_contracts(
        signal=1,
        confidence=0.2,
        net_liq=1_000_000,
        btc_price=75_000,
        target_gross_fraction=0.9,
        max_contracts=150,
        allow_short=True,
        min_confidence=0.1,
        current_contracts=-65,
    )
    assert order.desired_contracts > 0
    assert order.delta_contracts == order.desired_contracts + 65
    assert order.action == "buy"
