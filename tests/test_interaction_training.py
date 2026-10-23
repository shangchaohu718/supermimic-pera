"""Replay balance and all-task retention gates, using only tiny CPU modules."""
from collections import Counter
import copy
import importlib.util
from pathlib import Path

import pytest
import torch


_PATH = Path(__file__).resolve().parents[1] / "tools/train_interaction.py"
_SPEC = importlib.util.spec_from_file_location("interaction_training_test_target", _PATH)
training = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(training)


def test_replay_samples_every_task_equally_and_keeps_independent_history():
    replay = training.BalancedReplay(capacity=3)
    for task in range(len(training.TASKS)):
        replay.add(task, {"task": task, "version": "old"})
        replay.add(task, {"task": task, "version": "new"})
    counts = Counter(row["task"] for row in replay.sample(per_task=7))
    assert counts == {task: 7 for task in range(len(training.TASKS))}
    for task in range(len(training.TASKS)):
        assert list(replay.tasks[task]) == [
            {"task": task, "version": "old"}, {"task": task, "version": "new"},
        ]
    # Many new skateboard samples cannot evict other tasks' earlier examples.
    for index in range(20):
        replay.add(4, {"task": 4, "version": index})
    assert len(replay.tasks[4]) == 3
    assert [row["version"] for row in replay.tasks[4]] == [17, 18, 19]
    for task in range(4):
        assert replay.tasks[task][0]["version"] == "old"


def test_replay_refuses_training_when_any_task_has_no_real_experience():
    replay = training.BalancedReplay()
    for task in range(len(training.TASKS) - 1):
        replay.add(task, {"task": task})
    with pytest.raises(ValueError, match="Every task"):
        replay.sample()


class TinySharedPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.residual = torch.nn.Linear(64, 29)

    def forward(self, history, goal, context, mask):
        return self.residual(context) * mask[:, None]


def gate_fixture(tmp_path):
    # Intentionally bypass Training.__init__: no large checkpoints, simulator,
    # local datasets, CUDA context, or initialization rollout is required.
    learner = training.Training.__new__(training.Training)
    learner.out = tmp_path
    learner.iteration = 1
    learner.accepted = learner.rejected = 0
    learner.pi = TinySharedPolicy()
    with torch.no_grad():
        for parameter in learner.pi.parameters():
            parameter.fill_(.25)
    learner.stable_pi = copy.deepcopy(learner.pi.residual.state_dict())
    learner.opt_pi = torch.optim.Adam(learner.pi.parameters(), lr=.001)
    learner.pi.residual(torch.ones(1, 64)).sum().backward()
    learner.opt_pi.step()
    learner.opt_pi.zero_grad()
    assert learner.opt_pi.state  # A rejected update must clear stale Adam moments.
    learner.best_returns = {task: 100. for task in training.TASKS[1:]}
    learner.stable_metrics = {task: {"return": 100.} for task in training.TASKS[1:]}
    learner.tracking_retention = lambda: {"max_action_error": 0.}
    learner.memory_check = lambda: 0
    learner.batch = lambda: {"c": [0], "f": [0], "t": torch.tensor([1]),
                            "s": torch.zeros(1, 71), "o": torch.zeros(1, 27),
                            "h": torch.zeros(1, 10, 99)}
    learner.reference = lambda *args: (None, torch.zeros(1, 730), torch.zeros(1, 27))
    learner.context = lambda *args: torch.ones(1, 64)
    learner.saved = []

    def save(name):
        learner.saved.append(name)
        torch.save(learner.pi.residual.state_dict(), learner.out / name)

    learner.save = save
    (tmp_path / "accepted.pt").write_bytes(b"previous accepted checkpoint")
    return learner


@pytest.mark.parametrize("regressed_task", training.TASKS[1:])
def test_any_task_regression_rolls_back_actor_optimizer_and_preserves_checkpoint(tmp_path, regressed_task):
    learner = gate_fixture(tmp_path)
    metrics = {task: {"return": 110.} for task in training.TASKS[1:]}
    metrics[regressed_task] = {"return": 97.9}  # Below 2% allowance from 100.
    learner.evaluate = lambda: metrics
    with torch.no_grad():
        for parameter in learner.pi.parameters():
            parameter.fill_(.8)
    before = (tmp_path / "accepted.pt").read_bytes()
    assert learner.gate() is False
    assert learner.accepted == 0 and learner.rejected == 1
    assert not learner.opt_pi.state
    assert learner.saved == []
    assert (tmp_path / "accepted.pt").read_bytes() == before
    assert learner.best_returns == {task: 100. for task in training.TASKS[1:]}
    assert learner.stable_metrics == {task: {"return": 100.} for task in training.TASKS[1:]}
    for name, tensor in learner.pi.residual.state_dict().items():
        torch.testing.assert_close(tensor, learner.stable_pi[name], rtol=0, atol=0)
    # Replay anchor must use the rolled-back accepted actor, not the rejected one.
    history, goal, context, mask, target = learner.anchor
    torch.testing.assert_close(target, learner.pi(history, goal, context, mask), rtol=0, atol=0)


def test_retention_tolerance_never_ratchets_down_across_accepted_updates(tmp_path):
    learner = gate_fixture(tmp_path)
    learner.evaluate = lambda: {task: {"return": 99.} for task in training.TASKS[1:]}
    with torch.no_grad():
        for parameter in learner.pi.parameters():
            parameter.fill_(.6)
    assert learner.gate() is True
    assert learner.best_returns == {task: 100. for task in training.TASKS[1:]}
    accepted = (tmp_path / "accepted.pt").read_bytes()
    first_actor = copy.deepcopy(learner.pi.residual.state_dict())
    learner.iteration += 1
    # 97.1 is within 2% of the latest accepted 99, but below 98 from best=100.
    learner.evaluate = lambda: {task: {"return": 97.1} for task in training.TASKS[1:]}
    with torch.no_grad():
        for parameter in learner.pi.parameters():
            parameter.fill_(.9)
    assert learner.gate() is False
    assert learner.accepted == 1 and learner.rejected == 1
    assert learner.best_returns == {task: 100. for task in training.TASKS[1:]}
    assert learner.saved == ["accepted.pt"]
    assert (tmp_path / "accepted.pt").read_bytes() == accepted
    for name, tensor in learner.pi.residual.state_dict().items():
        torch.testing.assert_close(tensor, first_actor[name], rtol=0, atol=0)


def test_accepted_improvement_updates_each_task_best_and_checkpoint(tmp_path):
    learner = gate_fixture(tmp_path)
    metrics = {task: {"return": 100. + index} for index, task in enumerate(training.TASKS[1:], 1)}
    learner.evaluate = lambda: metrics
    assert learner.gate() is True
    assert learner.saved == ["accepted.pt"]
    assert learner.best_returns == {task: row["return"] for task, row in metrics.items()}
    assert learner.stable_metrics == metrics
    assert learner.opt_pi.state  # Accepted update keeps useful optimizer moments.
    saved = torch.load(tmp_path / "accepted.pt", weights_only=True)
    for name, tensor in learner.pi.residual.state_dict().items():
        torch.testing.assert_close(tensor, saved[name], rtol=0, atol=0)
