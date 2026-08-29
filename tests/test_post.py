import pytest
import torch

from facetorch.analyzer.predictor.post import (
    PostArgMax,
    PostSigmoidBinary,
    PostEmbedder,
    PostMultiLabel,
    PostLabelConfidencePairs,
)
from facetorch.datastruct import Prediction


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def labels_3():
    return ["A", "B", "C"]


@pytest.fixture
def labels_2():
    return ["Real", "Fake"]


@pytest.fixture
def post_argmax(device, labels_3):
    return PostArgMax(
        transform=None,
        device=device,
        optimize_transform=False,
        labels=labels_3,
        dim=1,
    )


@pytest.fixture
def post_sigmoid_binary(device, labels_2):
    return PostSigmoidBinary(
        transform=None,
        device=device,
        optimize_transform=False,
        labels=labels_2,
        threshold=0.5,
    )


@pytest.fixture
def post_embedder(device, labels_3):
    return PostEmbedder(
        transform=None,
        device=device,
        optimize_transform=False,
        labels=labels_3,
    )


@pytest.fixture
def post_multilabel(device, labels_3):
    return PostMultiLabel(
        transform=None,
        device=device,
        optimize_transform=False,
        labels=labels_3,
        dim=1,
        threshold=0.5,
    )


@pytest.fixture
def post_label_confidence(device, labels_3):
    return PostLabelConfidencePairs(
        transform=None,
        device=device,
        optimize_transform=False,
        labels=labels_3,
    )


class TestPostArgMax:
    @pytest.mark.unit
    def test_tensor_input(self, post_argmax):
        preds = torch.tensor([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])
        result = post_argmax.run(preds)
        assert len(result) == 2
        assert all(isinstance(p, Prediction) for p in result)
        assert result[0].label == "B"
        assert result[1].label == "A"

    @pytest.mark.unit
    def test_tuple_input(self, post_argmax):
        tensor = torch.tensor([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])
        preds = (tensor,)
        result = post_argmax.run(preds)
        assert len(result) == 2
        assert result[0].label == "B"
        assert result[1].label == "A"

    @pytest.mark.unit
    def test_tuple_multi_element(self, post_argmax):
        tensor = torch.tensor([[0.1, 0.8, 0.1]])
        extra = torch.tensor([[0.0, 0.0, 0.0]])
        preds = (tensor, extra)
        result = post_argmax.run(preds)
        assert len(result) == 1
        assert result[0].label == "B"


class TestPostSigmoidBinary:
    @pytest.mark.unit
    def test_tensor_input(self, post_sigmoid_binary):
        preds = torch.tensor([[3.0], [-3.0]])
        result = post_sigmoid_binary.run(preds)
        assert len(result) == 2
        assert all(isinstance(p, Prediction) for p in result)
        assert result[0].label == "Fake"
        assert result[1].label == "Real"

    @pytest.mark.unit
    def test_tuple_input(self, post_sigmoid_binary):
        tensor = torch.tensor([[3.0], [-3.0]])
        preds = (tensor,)
        result = post_sigmoid_binary.run(preds)
        assert len(result) == 2
        assert result[0].label == "Fake"
        assert result[1].label == "Real"

    @pytest.mark.unit
    def test_tuple_multi_element(self, post_sigmoid_binary):
        tensor = torch.tensor([[3.0]])
        extra = torch.tensor([[0.0]])
        preds = (tensor, extra)
        result = post_sigmoid_binary.run(preds)
        assert len(result) == 1
        assert result[0].label == "Fake"

    @pytest.mark.unit
    @pytest.mark.parametrize("threshold", (0.2, 0.5, 0.7))
    def test_threshold_is_inclusive_and_preserves_probability(
        self, device, labels_2, threshold
    ):
        processor = PostSigmoidBinary(
            transform=None,
            device=device,
            optimize_transform=False,
            labels=labels_2,
            threshold=threshold,
        )
        probabilities = torch.tensor(
            [threshold - 0.01, threshold, threshold + 0.01], dtype=torch.float64
        )
        logits = torch.logit(probabilities).unsqueeze(1)

        result = processor.run(logits)

        assert [prediction.label for prediction in result] == [
            "Real",
            "Fake",
            "Fake",
        ]
        torch.testing.assert_close(
            torch.stack([prediction.logits for prediction in result]),
            probabilities,
            rtol=0,
            atol=1e-15,
        )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "threshold",
        (True, False, -0.01, 1.01, float("nan"), float("inf"), "0.5"),
    )
    def test_rejects_invalid_thresholds(self, device, labels_2, threshold):
        with pytest.raises(ValueError, match="threshold"):
            PostSigmoidBinary(
                transform=None,
                device=device,
                optimize_transform=False,
                labels=labels_2,
                threshold=threshold,
            )

    @pytest.mark.unit
    def test_requires_exactly_two_labels(self, device):
        with pytest.raises(ValueError, match="two labels"):
            PostSigmoidBinary(
                transform=None,
                device=device,
                optimize_transform=False,
                labels=["only-one"],
            )


class TestPostEmbedder:
    @pytest.mark.unit
    def test_tensor_input(self, post_embedder):
        preds = torch.randn(2, 128)
        result = post_embedder.run(preds)
        assert len(result) == 2

    @pytest.mark.unit
    def test_tuple_input(self, post_embedder):
        tensor = torch.randn(2, 128)
        preds = (tensor,)
        result = post_embedder.run(preds)
        assert len(result) == 2


class TestPostMultiLabel:
    @pytest.mark.unit
    def test_tensor_input(self, post_multilabel):
        preds = torch.tensor([[0.9, 0.6, 0.1]])
        result = post_multilabel.run(preds)
        assert len(result) == 1
        assert result[0].label == "A"
        assert "multi" in result[0].other

    @pytest.mark.unit
    def test_tuple_input(self, post_multilabel):
        tensor = torch.tensor([[0.9, 0.6, 0.1]])
        preds = (tensor,)
        result = post_multilabel.run(preds)
        assert len(result) == 1
        assert result[0].label == "A"


class TestPostLabelConfidencePairs:
    @pytest.mark.unit
    def test_tensor_input(self, post_label_confidence):
        preds = torch.tensor([[0.5, 0.3, 0.2]])
        result = post_label_confidence.run(preds)
        assert len(result) == 1
        assert "A" in result[0].other

    @pytest.mark.unit
    def test_tuple_input(self, post_label_confidence):
        tensor = torch.tensor([[0.5, 0.3, 0.2]])
        preds = (tensor,)
        result = post_label_confidence.run(preds)
        assert len(result) == 1
        assert "A" in result[0].other
