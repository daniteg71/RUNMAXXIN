"""Test dello strato simbolico (SPARQL + SHACL): deterministici, solo rdflib/pyshacl."""
from symbolic import is_critical_state, validate_speed


def test_critical_state_below_threshold():
    assert is_critical_state(0.60) is False
    assert is_critical_state(0.86) is False


def test_critical_state_at_and_above_threshold():
    assert is_critical_state(0.90) is True
    assert is_critical_state(0.95) is True


def test_validate_speed_within_range():
    assert validate_speed(12.0) is True
    assert validate_speed(45.0) is True          # bordo incluso (sh:maxInclusive)


def test_validate_speed_out_of_range():
    assert validate_speed(45.1) is False
    assert validate_speed(300.0) is False
    assert validate_speed(-5.0) is False


def test_intent_drops_implausible_speed():
    from intent import parse_numbers
    assert "speed_kmh" not in parse_numbers("corro a 300 km/h")
    assert parse_numbers("corro a 12 km/h")["speed_kmh"] == 12.0
