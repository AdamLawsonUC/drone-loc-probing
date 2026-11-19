# Analysis: does the model's ego-pose estimate predict whether its commands succeed?

Added 19 Nov 2026 (Adam's proposal). The model is expected to be a poor pilot. That is useful: its mistakes can be
compared with what its activations say about where the drone is.

## The question

When the model's internal estimate of height (or camera pitch) is wrong, is its move wrong too, and wrong in the
direction the estimate predicts? If so, the estimate is being used to act, which is the link the steering
flights later test by intervention.

## What is measured

| Quantity | Source | Script |
| --- | --- | --- |
| True state: height, heading, camera pitch | drone telemetry in `commands.csv` | 61 |
| Task and its target (for example "about 80 m") | config in `session.json`; the target is not shown to the model as a number to hit by telemetry, only as words in the task | 63 |
| Move the model chose, and the move the envelope allowed | `chosen`, `action` in `commands.csv` | 63 |
| Correct move for the true state | from target and state | 63 |
| Task outcome: final error, time inside tolerance, success | per task attempt | 63 |
| Estimated state: the probe's held-out reading per frame | `predictions/*.npz` | 30 |

Three levels of "success of the movement":

1. Execution: did the drone do what was commanded (commanded speed against the change in height, heading, GPS).
   This is a property of the drone and the link, measured from the logs, and is the baseline.
2. Decision: was the chosen move the right one for the task at the true state.
3. Outcome: did the task end inside its tolerance.

## The tests

A. Correlation. Per frame, estimate error = |probe reading - true value|. Compare with decision correctness
   (logistic regression, with distance to target and task as covariates; Spearman at the task level between
   mean estimate error and final error).

B. Belief-consistent moves, the sharper test. For each frame compute two "correct" moves: one from the true
   state and one from the probe's reading. On frames where they differ, count how often the model's move matches
   the probe's version. Above chance means the model acts on the state its activations hold, not on the true
   state. Report with a bootstrap over task attempts.

C. Shuffle control. Repeat B with probe readings shuffled across frames of the same task, to show the effect is
   not produced by the task alone.

## Guard rails

- Use the image-token readout (`img_mean`) for the estimate. The image tokens come before the task text, so those
  activations cannot contain the task or the decision. The last-token readout can, and a probe there may simply be
  reading "about to say climb".
- Probe readings must be held-out: probes trained on other sites or other sessions than the frame they score.
- Mark only frames where the model was flying (`model_in_control`), and only tasks with a target.
- Decision correctness is about the chosen move, not the executed one: a move the envelope blocked still counts
  as the model's choice.
- This is correlational. A shared cause (a hard image degrades both the estimate and the move) is possible; test
  B's direction check narrows it, and the steering flight settles it.
- The altitude reading is height above take-off in 0.1 m steps; tolerances in the config are wider than that.

## Outputs for the paper

`results/<run>-command_scores/command_scores.csv`, `task_scores.csv`, `summary.csv` (script 63);
`results/<run>-probes/predictions/` (script 30); the joined analysis and its figure come from script 64, to be
written once the first model flights exist.
