"""Physical allocator snapshots restore dependencies, not display ordering."""
from dataclasses import asdict
import copy
import json

import pytest

from heterollm_sim.memory_allocator import AllocationError, PhysicalAddressAllocator


@pytest.mark.parametrize("representation", ["typed", "json", "mixed"])
def test_snapshot_restores_alias_chain_that_sorts_before_its_target(representation):
    allocator = PhysicalAddressAllocator(32768, workspace_capacity_bytes=4096)
    allocator.allocate("zz-root", 8192, 4, allocation_arena="resident")
    allocator.allocate("bb-view", 256, 7, alias_of="zz-root", alias_generation=4)
    allocator.allocate("aa-nested", 64, 9, alias_of="bb-view", alias_generation=7)
    allocator.allocate("workspace", 256, 0, allocation_arena="workspace")
    saved = allocator.snapshot()
    assert [a.buffer_id for a in saved["allocations"]][:3] == ["aa-nested", "bb-view", "zz-root"]
    if representation == "json":
        saved["allocations"] = [asdict(item) for item in saved["allocations"]]
        saved = json.loads(json.dumps(saved))
    elif representation == "mixed":
        saved["allocations"] = tuple(asdict(item) if index % 2 else item
                                     for index, item in enumerate(saved["allocations"]))
    before = copy.deepcopy(saved)
    for _ in range(2):
        restored = PhysicalAddressAllocator.from_snapshot(saved)
        assert restored.snapshot() == allocator.snapshot()
        assert restored.lookup("aa-nested", 9).allocation_arena == "resident"
        assert restored.lookup("workspace").base_address == 28672
        assert restored.address("aa-nested", 0, 64, 9) == 0
        with pytest.raises(AllocationError, match="live aliases"):
            restored.release("zz-root", 4)
        restored.release("aa-nested", 9)
        restored.release("bb-view", 7)
        restored.release("zz-root", 4)
        assert restored.allocate("new", 64).base_address == 0
    assert saved == before


@pytest.mark.parametrize("aliases", [
    {"a": "absent"}, {"a": "a"}, {"a": "b", "b": "a"},
])
def test_json_snapshot_rejects_missing_targets_and_cycles(aliases):
    saved = {"capacity_bytes": 4096, "alignment_bytes": 64, "allocations": [
        {"buffer_id": name, "base_address": 0, "size_bytes": 64,
         "generation": 3, "alias_of": target, "alias_generation": 3,
         "allocation_arena": "resident"}
        for name, target in aliases.items()
    ]}
    with pytest.raises(AllocationError, match="target is missing|cycle"):
        PhysicalAddressAllocator.from_snapshot(saved)


def test_json_snapshot_resolves_alias_generation_independently_of_name():
    allocator = PhysicalAddressAllocator(4096)
    allocator.allocate("z", 64, 1, allocation_arena="resident")
    allocator.allocate("z", 64, 2, allocation_arena="resident")
    allocator.allocate("a", 32, 3, alias_of="z", alias_generation=2)
    saved = allocator.snapshot()
    saved["allocations"] = [asdict(item) for item in reversed(saved["allocations"])]
    restored = PhysicalAddressAllocator.from_snapshot(saved)
    assert restored.address("a", 0, 32, 3) == 64
    assert restored.lookup("a", 3).alias_generation == 2
