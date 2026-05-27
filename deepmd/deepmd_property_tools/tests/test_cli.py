# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

from pathlib import (
    Path,
)
from unittest import (
    mock,
)

from deepmd_property_tools import (
    cli,
)


def test_main_prints_help_without_command(capsys) -> None:
    exit_code = cli.main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "DeePMD molecular property training" in captured.out


def test_train_command_calls_property_train() -> None:
    trainer = mock.Mock()
    with mock.patch.object(cli, "PropertyTrain", return_value=trainer) as train_cls:
        exit_code = cli.main(
            [
                "train",
                "--dataset",
                "data.csv",
                "--mol-dir",
                "mol",
                "--save-path",
                "exp",
                "--numb-steps",
                "10",
                "--batch-size",
                "1",
            ]
        )

    assert exit_code == 0
    train_cls.assert_called_once()
    trainer.fit.assert_called_once_with(
        {"dataset": Path("data.csv"), "mol_dir": Path("mol")}
    )


def test_predict_command_calls_property_predict() -> None:
    predictor = mock.Mock()
    predictor.predict.return_value = [[1.0]]
    with mock.patch.object(cli, "PropertyPredict", return_value=predictor):
        with mock.patch("builtins.print"):
            exit_code = cli.main(
                [
                    "predict",
                    "--model",
                    "exp",
                    "--dataset",
                    "data.csv",
                    "--mol-dir",
                    "mol",
                    "--save-path",
                    "pred",
                ]
            )

    assert exit_code == 0
    predictor.predict.assert_called_once_with(
        {"dataset": Path("data.csv"), "mol_dir": Path("mol")},
        save_path=Path("pred"),
    )
