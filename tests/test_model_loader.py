from app.model_loader import validate_model_contract


class FakeModel:
    def __init__(self, shape):
        self.output_shape = shape


def test_model_contract_accepts_expected_output():
    validate_model_contract(FakeModel((None, 7)), 7)


def test_model_contract_rejects_wrong_class_count():
    import pytest

    with pytest.raises(ValueError, match="incompatible"):
        validate_model_contract(FakeModel((None, 8)), 7)
