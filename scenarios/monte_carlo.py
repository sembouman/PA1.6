import argparse
import csv
import os
from statistics import fmean, pstdev
from typing import Optional

from controllers.onoff import OnOffThermostat
from controllers.predictive_onoff import PredictiveOnOff
from sensors.filters import MovingAverageFilter, hold_last
from sensors.temp_sensor import TempSensor
from simulations.environment import Environment
from simulations.room_model import step_room
from utils.config import load_config
from utils.rng import RNG


def run_monte_carlo(
    scenario_path: str,
    runs: int = 100,
    seed_start: int = 0,
    output_path: str = "outputs/logs/monte_carlo.csv",
) -> list[dict[str, int | float]]:
    """Run a scenario with consecutive seeds and save per-run regulation metrics."""
    if runs < 1:
        raise ValueError("runs must be at least 1")

    scenario = load_config(scenario_path)
    dt = scenario.sim.dt
    if dt <= 0:
        raise ValueError("simulation dt must be positive")
    steps = int(scenario.sim.duration_s / dt)
    if steps < 1:
        raise ValueError("simulation duration must be at least one time step")

    results: list[dict[str, int | float]] = []
    for run_index in range(runs):
        seed = seed_start + run_index
        rng = RNG(seed)
        env = Environment(**vars(scenario.env))
        sensor = TempSensor(**vars(scenario.sensor), rng=rng)
        if scenario.controller.type == "onoff":
            controller = OnOffThermostat(
                setpoint=scenario.controller.setpoint,
                deadband=scenario.controller.deadband,
                safety_high=scenario.controller.safety_high,
            )
            predictive = False
        elif scenario.controller.type == "predictive_onoff":
            controller = PredictiveOnOff(
                setpoint=scenario.controller.setpoint,
                deadband=scenario.controller.deadband,
                tau=scenario.controller.tau,
                safety_high=scenario.controller.safety_high,
            )
            predictive = True
        else:
            raise ValueError(f"Unknown controller type: {scenario.controller.type}")

        temperature = scenario.sim.init_T
        last_valid: Optional[float] = temperature
        moving_average = MovingAverageFilter(window=5)
        absolute_errors: list[float] = []
        squared_errors: list[float] = []
        heater_on_steps = 0
        maximum_temperature = temperature

        for step in range(steps):
            time_s = step * dt
            measurement = hold_last(sensor.read(temperature), last_valid)
            if measurement is None:
                measurement = temperature
            last_valid = measurement
            filtered = moving_average.update(measurement)

            if predictive:
                heater = controller.update(filtered, dt)
            else:
                heater = controller.update(filtered)

            error = temperature - controller.setpoint
            absolute_errors.append(abs(error))
            squared_errors.append(error * error)
            heater_on_steps += heater
            maximum_temperature = max(maximum_temperature, temperature)

            temperature = step_room(
                temperature,
                heater,
                env.T_out(time_s),
                scenario.model.R,
                scenario.model.C,
                scenario.model.P,
                dt,
                scenario.model.process_sigma,
                rng,
            )

        results.append({
            "seed": seed,
            "mean_abs_error_C": fmean(absolute_errors),
            "rmse_C": (fmean(squared_errors)) ** 0.5,
            "duty_cycle": heater_on_steps / steps,
            "max_temperature_C": maximum_temperature,
        })

    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate thermostat performance across random seeds")
    parser.add_argument("--scenario", required=True, help="Path to a YAML scenario file")
    parser.add_argument("--runs", type=int, default=100, help="Number of simulations to run")
    parser.add_argument("--seed-start", type=int, default=0, help="First random seed")
    parser.add_argument(
        "--output",
        default="outputs/logs/monte_carlo.csv",
        help="CSV path for per-run metrics",
    )
    args = parser.parse_args()

    results = run_monte_carlo(args.scenario, args.runs, args.seed_start, args.output)
    metrics = ("mean_abs_error_C", "rmse_C", "duty_cycle", "max_temperature_C")
    print(f"Completed {len(results)} runs; per-run metrics saved to {args.output}")
    for metric in metrics:
        values = [float(result[metric]) for result in results]
        print(f"{metric}: mean={fmean(values):.4f}, standard deviation={pstdev(values):.4f}")


if __name__ == "__main__":
    main()