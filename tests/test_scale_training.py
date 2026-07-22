import pytest

from textjepa.data.sampling import DistributedFreshEpochSampler
from textjepa.training.scale import crossed_milestones, optimizer_steps


class Sized:
    def __len__(self):
        return 12


def test_distributed_fresh_sampler_is_disjoint_complete_and_fresh():
    samplers = [DistributedFreshEpochSampler(Sized(), rank, 3, seed=7)
                for rank in range(3)]
    first = [set(sampler) for sampler in samplers]
    assert set.union(*first) == set(range(12))
    assert not any(first[i] & first[j] for i in range(3) for j in range(i))
    for sampler in samplers:
        sampler.set_epoch(1)
    second = [set(sampler) for sampler in samplers]
    assert set.union(*second) == set(range(12, 24))


def test_distributed_sampler_rejects_uneven_dataset():
    with pytest.raises(ValueError, match="divisible"):
        DistributedFreshEpochSampler(range(10), 0, 3)


def test_scale_accounting_and_milestones_are_exact():
    assert optimizer_steps(128, 8) == 16
    with pytest.raises(ValueError):
        optimizer_steps(127, 8)
    assert crossed_milestones(900, 5100, (1000, 5000, 10000)) == (1000, 5000)
