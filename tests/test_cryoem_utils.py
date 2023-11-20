import os
import pytest
import pandas as pd
from utils import cryoem_utils as cu

FILE_PATH = os.path.dirname(os.path.abspath(__file__))


def test_empty_extract_qscores():
    df = cu.extract_qscores(f"{FILE_PATH}/data/Empty_QScores.txt")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_nag_extract_qscores():
    df = cu.extract_qscores(f"{FILE_PATH}/data/NAG_QScores.txt")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == 10
    assert df["chain"].tolist() == ["I", "I", "H", "H", "J", "J", "G", "G", "F", "F"]
    assert df["res_id"].tolist() == ["1", "2", "1", "2", "1", "2", "1", "2", "1", "2"]
    assert df["res_name"].tolist() == ["NAG"] * 10

    obtained_qscores = df["qscore"].tolist()
    expected_qscores = [0.55, 0.46, 0.52, 0.49, 0.35, 0.41, 0.52, 0.47, 0.53, 0.47]
    assert obtained_qscores == pytest.approx(expected_qscores, abs=0.01)


def test_dtp_extract_qscores():
    df = cu.extract_qscores(f"{FILE_PATH}/data/DTP_QScores.txt")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == 15
    chains_1 = ["C", "E", "D", "G", "F", "I", "H"]
    chains_2 = ["K", "J", "M", "L", "O", "N", "Q", "P"]
    assert df["chain"].tolist() == chains_1 + chains_2
    assert df["res_id"].tolist() == ["1301"] * 15
    assert df["res_name"].tolist() == ["DTP"] * 15

    obtained_qscores = df["qscore"].tolist()
    qscores_1 = [0.55, 0.54, 0.58, 0.57, 0.59, 0.57, 0.56]
    qscores_2 = [0.55, 0.57, 0.57, 0.55, 0.55, 0.56, 0.53, 0.56]
    expected_qscores = qscores_1 + qscores_2
    assert obtained_qscores == pytest.approx(expected_qscores, abs=0.01)


def test_create_ligand_only_pdb():
    cif_file = "tests/data/8sor.cif"
    output_file = "tests/data/test_ligand.pdb"
    command_file = "tests/data/chimera_command.py"
    chimera_dir = "C:/Tools/Crystallography/Chimera 1.17.3"

    cu.create_ligand_only_pdb(cif_file, output_file, command_file, chimera_dir)

    assert os.path.exists(output_file)
    os.remove(output_file)
