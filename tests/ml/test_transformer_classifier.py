import torch
from torch import nn

from nlp_platform.ml.transformer_classifier import TransformerClassifier
from nlp_platform.ml.transformer_config import TransformerConfig


class FakeOutput:
    def __init__(self) -> None:
        self.logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])


class FakeTransformerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(4, 4)
        self.last_input_ids: torch.Tensor | None = None
        self.last_attention_mask: torch.Tensor | None = None
        self.last_labels: torch.Tensor | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> FakeOutput:
        self.last_input_ids = input_ids
        self.last_attention_mask = attention_mask
        self.last_labels = labels
        return FakeOutput()


def test_forward_passes_inputs_to_model() -> None:
    model = FakeTransformerModel()
    classifier = TransformerClassifier(model)

    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.tensor([[1, 1, 1, 1]])
    labels = torch.tensor([2])

    output = classifier.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    assert torch.equal(model.last_input_ids, input_ids)
    assert torch.equal(model.last_attention_mask, attention_mask)
    assert torch.equal(model.last_labels, labels)
    assert output.logits.shape == (1, 4)


def test_forward_supports_missing_labels() -> None:
    model = FakeTransformerModel()
    classifier = TransformerClassifier(model)

    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.tensor([[1, 1, 1, 1]])

    classifier.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    assert model.last_labels is None


def test_parameters_returns_model_parameters() -> None:
    model = FakeTransformerModel()
    classifier = TransformerClassifier(model)

    parameters = list(classifier.parameters())

    assert len(parameters) == 2


def test_train_sets_training_mode() -> None:
    model = FakeTransformerModel()
    classifier = TransformerClassifier(model)

    classifier.eval()
    assert not model.training

    classifier.train()

    assert model.training


def test_eval_sets_evaluation_mode() -> None:
    model = FakeTransformerModel()
    classifier = TransformerClassifier(model)

    classifier.train()
    assert model.training

    classifier.eval()

    assert not model.training


def test_to_returns_classifier() -> None:
    model = FakeTransformerModel()
    classifier = TransformerClassifier(model)

    result = classifier.to(torch.device("cpu"))

    assert result is classifier


def test_model_property_returns_underlying_model() -> None:
    model = FakeTransformerModel()
    classifier = TransformerClassifier(model)

    assert classifier.model is model


def test_config_property_returns_configuration() -> None:
    config = TransformerConfig(
        model_name="test-model",
        max_length=64,
        num_labels=3,
    )
    classifier = TransformerClassifier(
        model=FakeTransformerModel(),
        config=config,
    )

    assert classifier.config == config