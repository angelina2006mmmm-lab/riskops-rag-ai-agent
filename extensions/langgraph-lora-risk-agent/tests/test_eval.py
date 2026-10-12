from scripts.evaluate import run

def test_small_synthetic_benchmark():
 r=run()
 assert r["cases"]==10
 assert r["status_accuracy"]==1
 assert r["citation_recall"]==1
 assert r["failures"]==[]
