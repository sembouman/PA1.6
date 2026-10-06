import csv

from scenarios.monte_carlo import run_monte_carlo


def test_monte_carlo_is_reproducible_and_writes_metrics(tmp_path):
    scenario = "scenarios/cold_morning.yaml"
    first_output = tmp_path / "first.csv"
    second_output = tmp_path / "second.csv"

    first = run_monte_carlo(scenario, runs=2, seed_start=40, output_path=str(first_output))
    second = run_monte_carlo(scenario, runs=2, seed_start=40, output_path=str(second_output))

    assert first == second
    assert [result["seed"] for result in first] == [40, 41]
    with first_output.open(newline="", encoding="utf-8") as output_file:
        rows = list(csv.DictReader(output_file))
    assert len(rows) == 2
    assert 0.0 <= float(rows[0]["duty_cycle"]) <= 1.0