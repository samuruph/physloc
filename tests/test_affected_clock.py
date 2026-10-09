"""The dense `affected` clock and the violator records it is rebuilt from must
translate ids the same way. They did not: on `shadow_track` multi a peer
violator that disturbed the actor's shadow was refused at write time
("affected clock disagrees with the violator records")."""
import numpy as np

from physloc import loader
from physloc.schema import write

ACTOR, PEER, SHADOW, T = 3, 5, 9, 12
REMAP = {SHADOW: ACTOR}                 # the shadow is published as its caster


def test_shadow_maps_to_its_caster_and_a_violator_never_affects_itself():
    record = {"affected_instance_ids": [SHADOW, PEER]}
    assert write.affected_ids(record, PEER, REMAP) == [ACTOR]
    assert write.affected_ids(record, ACTOR, REMAP) == [PEER]


def test_a_peer_disturbing_the_actors_shadow_passes_the_write_check():
    consequence = [[4, 9]]
    generated = {"affected_instance_ids": [SHADOW], "consequence_windows": consequence}
    ids = [ACTOR, PEER]
    row = {i: r for r, i in enumerate(ids)}
    clock = lambda windows: np.stack([np.zeros(T, bool), loader.intervals_to_mask(windows, T)])
    objects = {"ids": np.asarray(ids), "is_violator": np.asarray([False, True]),
               "active": clock([[4, 4]]), "intervening": clock([[4, 4]]),
               "consequence": clock(consequence), "observable": clock([[4, 11]]),
               "occluded": np.zeros((2, T), bool), "affected": np.zeros((2, T), bool)}
    # ...exactly as `annotate_pair` fills it
    for target in write.affected_ids(generated, PEER, REMAP):
        for start, end in consequence:
            objects["affected"][row[target], start:end + 1] = True
    document = {"violation": {"violators": [{
        "id": PEER, "affected_ids": write.affected_ids(generated, PEER, REMAP),
        "windows": {"active": [[4, 4]], "intervening": [[4, 4]],
                    "consequence": consequence, "observable": [[4, 11]]}}]}}
    write._record_clocks(document, objects, row, T)
