import numpy as np

from recipe.exact.core_exact import IdentityPotential, build_scoped_conserved_atoms
from recipe.exact.credit_spec import FactorSnapshot
from recipe.exact.env_probes import alfworld_intermediate_reward_snapshot


def _typed_snapshot(checkpoint_id, cumulative):
    record = alfworld_intermediate_reward_snapshot(cumulative)
    return FactorSnapshot(
        checkpoint_id=checkpoint_id,
        factor_ids=record["factor_ids"],
        values=np.asarray(record["values"], dtype=np.float64),
        read_sets=record["read_sets"],
        schema_version=record["schema_version"],
        channel_roles=record["channel_roles"],
        potential_weights=np.asarray(record["potential_weights"], dtype=np.float64),
        source_revision=record["source_revision"],
    )


def test_official_step_signals_are_checkpoint_differences():
    official_signals = (1.0, 0.0, -1.0, 1.0)
    cumulative = np.cumsum((0.0, *official_signals))
    snapshots = [_typed_snapshot(index, value) for index, value in enumerate(cumulative)]

    observed = np.diff([snapshot.values[0] for snapshot in snapshots])

    np.testing.assert_array_equal(observed, official_signals)


def test_process_channel_is_zero_target_and_binary_return_stays_opaque():
    snapshots = (
        _typed_snapshot(0, 0.0),
        _typed_snapshot(1, 1.0),
        _typed_snapshot(2, 1.0),
    )
    potential = IdentityPotential(snapshots[0].factor_ids, np.asarray([1.0]))

    conserved = build_scoped_conserved_atoms(snapshots, 10.0, potential)

    channel_id = "textworld_intermediate_reward_cumulative"
    assert conserved.channel_target_masses == {channel_id: 0.0}
    assert conserved.opaque_target_atom.value == 10.0
    assert sum(atom.value for atom in conserved.atoms) == 10.0
