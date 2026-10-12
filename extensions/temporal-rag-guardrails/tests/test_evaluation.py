from temporal_guard.evaluate import evaluate
def test_cases():
    x=evaluate()
    assert x["cases"]==12 and x["passed"]==12, x
