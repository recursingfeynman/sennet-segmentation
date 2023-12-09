import numpy as np
import pytest
import torch

from angionet.losses import ComboLoss, DiceLoss, GenSurfLoss
from angionet.losses._base import BaseLoss


class TestBaseLoss:
    @pytest.fixture
    def class_weights(self):
        y_true = torch.tensor([[[[1, 0, 1], [0, 1, 1]], [[0, 0, 1], [0, 0, 1]]]])
        class_weights = BaseLoss().compute_weights(y_true).numpy().ravel()
        return class_weights

    def test_class_weights_range(self, class_weights):
        for weight in class_weights:
            assert 0.0 <= weight <= 1.0

    def test_class_weights_sum(self, class_weights):
        assert np.sum(class_weights) == pytest.approx(1.0, 1e-6)

    def test_class_weights_values(self, class_weights):
        expected = [0.333, 0.667]
        assert np.allclose(expected, class_weights, atol=1e-3)


class TestDiceLoss:
    @pytest.mark.parametrize(
        ["y_pred", "y_true", "expected", "eps"],
        [
            ([[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]], 0.0, 1e-6),  # Ideal
            ([[0, 0, 0], [0, 0, 0]], [[1, 1, 1], [1, 1, 1]], 1.0, 1e-6),  # Worst
            ([[1, 0, 1], [0, 1, 0]], [[0, 1, 0], [0, 1, 0]], 0.6, 1e-6),  # Random
        ],
    )
    def test_results(self, y_pred, y_true, expected, eps):
        y_pred = torch.tensor(y_pred).view(1, 1, 1, -1).float()
        y_true = torch.tensor(y_true).view(1, 1, 1, -1).float()

        criterion = DiceLoss(sigmoid=False)
        loss = criterion(y_pred, y_true)

        assert float(loss) == pytest.approx(expected, abs=eps)

    def test_logits_as_input(self):
        input_good = torch.tensor([[10.0, -5.0, 10.0]]).view(1, 1, 1, -1)
        input_bad = torch.tensor([[2.0, -1.0, 2.0]]).view(1, 1, 1, -1)

        y_true = torch.tensor([[1, 0, 1]]).view(1, 1, 1, -1)
        criterion = DiceLoss(sigmoid=True)
        loss_good = criterion(input_good, y_true)
        loss_bad = criterion(input_bad, y_true)

        assert loss_bad > loss_good


class TestGenSurfLoss:
    @pytest.mark.parametrize(["steps", "expected"], [(3, 3), (5, 5), (0, 0), (-2, 0)])
    def test_alpha_count(self, steps, expected):
        criterion = GenSurfLoss(None, steps)

        alphas = []
        for _ in range(steps):
            alphas += [criterion.alpha]
            criterion.step()

        assert len(alphas) == expected

    @pytest.fixture
    def alpha(self):
        steps = 10
        criterion = GenSurfLoss(None, steps)

        alphas = []
        for _ in range(steps):
            alphas += [round(criterion.alpha, 2)]
            criterion.step()

        return alphas

    def test_first_alpha_is_one(self, alpha):
        assert alpha[0] == pytest.approx(1.0, abs=1e-6)

    def test_last_alpha_is_zero(self, alpha):
        assert alpha[-1] == pytest.approx(0.0, abs=1e-6)

    def test_alpha_slope_is_negative(self, alpha):
        slope = np.diff(alpha)
        assert (slope < 0).all()

    def test_cosine_scheduler_values(self, alpha):
        expected = [1.0, 0.97, 0.88, 0.75, 0.59, 0.41, 0.25, 0.12, 0.03, 0.0]
        assert np.allclose(alpha, expected, atol=1e-2)

    @pytest.mark.parametrize(
        ["y_pred", "y_true", "dtm", "expected", "eps"],
        [
            (
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
                0.0,
                1e-6,
            ),  # Ideal
            (
                [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
                1.0,
                1e-6,
            ),  # Worst
            (
                [[0, 1, 1], [1, 1, 0], [1, 1, 1]],
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
                0.8,
                1e-6,
            ),  # Random
        ],
    )
    def test_results(self, y_pred, y_true, dtm, expected, eps):
        y_pred = torch.tensor(y_pred).view(1, 1, 3, 3).float()
        y_true = torch.tensor(y_true).view(1, 1, 3, 3).float()
        dtm = torch.tensor(dtm).view(1, 1, 3, 3).float()

        criterion = GenSurfLoss(
            region_loss=DiceLoss(sigmoid=False), total_steps=1, sigmoid=False
        )
        criterion.alpha = 0
        loss = criterion(y_pred, y_true, dtm)

        assert loss == pytest.approx(expected, abs=eps)

    def test_logits_as_input(self):
        input_good = torch.tensor(
            [[-5.0, 10.0, -5.0], [10.0, -5.0, 10.0], [-5.0, 10.0, -5.0]]
        ).view(1, 1, 1, -1)
        input_bad = torch.tensor(
            [[2.0, -1.0, 2.0], [2.0, -1.0, 2.0], [2.0, -1.0, 2.0]]
        ).view(1, 1, 1, -1)

        y_true = (
            torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]]).view(1, 1, 1, -1).float()
        )
        dtm = (
            torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
            .view(1, 1, 1, -1)
            .float()
        )
        criterion = GenSurfLoss(DiceLoss(sigmoid=True), 1, sigmoid=True)
        criterion.alpha = 0
        loss_good = criterion(input_good, y_true, dtm)
        loss_bad = criterion(input_bad, y_true, dtm)

        assert loss_bad > loss_good

    def test_logits_as_input_with_alpha(self):
        input_good = torch.tensor(
            [[-5.0, 10.0, -5.0], [10.0, -5.0, 10.0], [-5.0, 10.0, -5.0]]
        ).view(1, 1, 1, -1)
        input_bad = torch.tensor(
            [[2.0, -1.0, 2.0], [2.0, -1.0, 2.0], [2.0, -1.0, 2.0]]
        ).view(1, 1, 1, -1)

        y_true = (
            torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]]).view(1, 1, 1, -1).float()
        )
        dtm = (
            torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
            .view(1, 1, 1, -1)
            .float()
        )

        total_steps = 10
        criterion = GenSurfLoss(DiceLoss(), total_steps=total_steps)
        for _ in range(total_steps):
            loss_good = criterion(input_good, y_true, dtm)
            loss_bad = criterion(input_bad, y_true, dtm)

            assert loss_good < loss_bad

            criterion.step()


class TestComboLoss:
    def test_requires_grad(self):
        y_pred = torch.tensor([[1, 1, 1], [1, 1, 1]]).view(1, 1, 1, -1).float()
        y_true = torch.tensor([[1, 1, 1], [1, 1, 1]]).view(1, 1, 1, -1).float()
        criterion = ComboLoss([torch.nn.BCELoss()])
        loss = criterion(y_pred, y_true)

        assert loss.requires_grad_

    @pytest.mark.parametrize(
        ["y_pred", "y_true", "expected", "eps"],
        [
            ([[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]], 0.0, 1e-6),
            ([[0, 0, 0], [0, 0, 0]], [[1, 1, 1], [1, 1, 1]], 100.0, 1e-6),
            ([[1, 0, 1], [0, 1, 0]], [[0, 1, 0], [0, 1, 0]], 50.0, 1e-6),
        ],
    )
    def test_results_is_combination(self, y_pred, y_true, expected, eps):
        y_pred = torch.tensor(y_pred).view(1, 1, 1, -1).float()
        y_true = torch.tensor(y_true).view(1, 1, 1, -1).float()

        criterion_1 = torch.nn.BCELoss()
        criterion_2 = torch.nn.BCELoss()

        criterion = ComboLoss([criterion_1, criterion_2])

        loss = criterion(y_pred, y_true) / 2.0

        assert loss.detach().item() == pytest.approx(expected, abs=eps)
