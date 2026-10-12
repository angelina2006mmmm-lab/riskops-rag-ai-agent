import pytest
def test_smoke():
 pytest.importorskip("torch")
 from training.lora_smoke import run_smoke
 before,after=run_smoke()
 assert after<before
