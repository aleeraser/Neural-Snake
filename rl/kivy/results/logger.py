import csv
import os
from collections import deque
from statistics import mean

import matplotlib

matplotlib.use('Agg')

SCORES_CSV_PATH = "./results/results.csv"
SOLVED_CSV_PATH = "./results/solved.csv"
AVERAGE_SCORE_TO_SOLVE = 100
CONSECUTIVE_RUNS_TO_SOLVE = 50


class Logger:

    def __init__(self, env_name):
        self.steps = deque(maxlen=CONSECUTIVE_RUNS_TO_SOLVE)
        self.env_name = env_name

        if os.path.exists(SCORES_CSV_PATH):
            os.remove(SCORES_CSV_PATH)

    def log(self, step, run):
        self._save_csv(SCORES_CSV_PATH, step)
        self.steps.append(step)
        mean_step = mean(self.steps)
        print("Scores: (min: " + str(min(self.steps)) + ", avg: " + str(mean_step) + ", max: " + str(max(self.steps)) + ")\n")
        if mean_step >= AVERAGE_SCORE_TO_SOLVE and len(self.steps) >= CONSECUTIVE_RUNS_TO_SOLVE:
            solve_step = run - CONSECUTIVE_RUNS_TO_SOLVE
            print("Solved in " + str(solve_step) + " runs, " + str(run) + " total runs.")
            self._save_csv(SOLVED_CSV_PATH, solve_step)
            exit()

    def _save_csv(self, path, step):
        if not os.path.exists(path):
            with open(path, "w"):
                pass
        steps_file = open(path, "a")
        with steps_file:
            writer = csv.writer(steps_file)
            writer.writerow([step])
