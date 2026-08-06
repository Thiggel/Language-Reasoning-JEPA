from textjepa.data.sampling import FreshEpochSampler, GroupedTrajectoryBatchSampler


class _Data:
    def __init__(self, n: int):
        self.n = n

    def __len__(self):
        return self.n


def test_fresh_epoch_sampler_resume_is_exact_suffix():
    sampler = FreshEpochSampler(_Data(23), seed=17)
    sampler.set_epoch(4)
    full = list(sampler)
    sampler.set_start(9)
    assert list(sampler) == full[9:]


def test_grouped_sampler_resume_is_exact_suffix():
    sampler = GroupedTrajectoryBatchSampler(
        base_size=12, variants=2, bases_per_batch=3, microbatch_size=2, seed=7
    )
    sampler.set_epoch(2)
    full = list(sampler)
    sampler.set_start(4)
    assert list(sampler) == full[4:]
