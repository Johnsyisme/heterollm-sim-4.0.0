"""Explicit bank-group timing inputs remain optional, typed, and editable."""
from dataclasses import asdict, fields, replace
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from heterollm_sim.memory_types import DRAM_BANK_GROUP_TIMING_FIELDS, DramConfig
from heterollm_sim.physical_contract import (
    DRAM_OPTIONAL_BANK_GROUP_TIMING_FIELDS, parse_execution_physical_config,
    parse_required_physical_config,
)


@pytest.mark.parametrize("kind", ["DDR", "LPDDR", "HBM", "GDDR"])
def test_old_complete_mapping_omits_extension_without_inferred_timings(kind):
    original = DramConfig(kind=kind)
    raw = asdict(original)
    for name in DRAM_BANK_GROUP_TIMING_FIELDS:
        raw.pop(name)
    parsed = parse_required_physical_config("memory", kind.lower(), raw)
    assert not parsed.bank_group_timing_enabled
    assert all(getattr(parsed, name) == 0. for name in DRAM_BANK_GROUP_TIMING_FIELDS)
    assert parsed == original


@pytest.mark.parametrize("name", DRAM_BANK_GROUP_TIMING_FIELDS)
def test_individual_constraint_enables_and_roundtrips_without_changing_geometry(name):
    original = DramConfig()
    config = replace(original, **{name: 7.25})
    assert config.bank_group_timing_enabled
    assert config.capacity_bytes == original.capacity_bytes
    assert config.bank_count == original.bank_count
    assert config.lane_count == original.lane_count
    assert config.physical_interface_bandwidth_gb_s == original.physical_interface_bandwidth_gb_s
    raw = json.loads(json.dumps(asdict(config)))
    assert parse_execution_physical_config(raw) == config
    assert not replace(config, **{name: 0.}).bank_group_timing_enabled


@pytest.mark.parametrize("disabled", DRAM_BANK_GROUP_TIMING_FIELDS)
def test_zero_disables_one_constraint_without_replacing_other_authored_values(disabled):
    values = {name: 7.25 for name in DRAM_BANK_GROUP_TIMING_FIELDS}
    values[disabled] = 0.
    parsed = parse_execution_physical_config({**asdict(DramConfig()), **values})
    assert parsed.bank_group_timing_enabled
    assert {name: getattr(parsed, name) for name in DRAM_BANK_GROUP_TIMING_FIELDS} == values


@pytest.mark.parametrize("name", DRAM_BANK_GROUP_TIMING_FIELDS)
@pytest.mark.parametrize("value", [-1., float("inf"), float("nan"), True, "1", None])
def test_group_timings_reject_invalid_numeric_contract(name, value):
    raw = asdict(DramConfig())
    raw[name] = value
    with pytest.raises(ValueError, match=name):
        parse_execution_physical_config(raw)


def test_existing_positional_constructor_keeps_metadata_in_the_same_slot():
    original = DramConfig(metadata={"explicit": "old positional configuration"})
    declared = fields(DramConfig)
    metadata_index = next(index for index, item in enumerate(declared) if item.name == "metadata")
    old_arguments = [getattr(original, item.name) for item in declared[:metadata_index + 1]]
    restored = DramConfig(*old_arguments)
    assert restored.metadata == original.metadata
    assert not restored.bank_group_timing_enabled
    assert DRAM_OPTIONAL_BANK_GROUP_TIMING_FIELDS == frozenset(DRAM_BANK_GROUP_TIMING_FIELDS)


def test_browser_validation_and_editor_preserve_optional_timing_contract():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is needed for the browser configuration contract")
    path = Path(__file__).resolve().parents[1] / "src/heterollm_sim/webui/app.js"
    source = path.read_text(encoding="utf-8")
    # Run the actual pure validator/editor functions without starting the app.
    start = source.index("const PHYSICAL_MEMORY_COMPONENT_KINDS")
    end = source.index("function missingScenarioInputIssues", start)
    config = asdict(DramConfig())
    config["kind"] = "DDR"
    for name in DRAM_BANK_GROUP_TIMING_FIELDS:
        config.pop(name)
    script = """
const assert = require('node:assert/strict');
const normalizedComponentKind = value => String(value).toLowerCase();
const uiText = zh => zh;
const escapeHtml = value => String(value);
""" + source[start:end] + "\nconst raw = " + json.dumps(config) + ";\n" + """
const component = {kind: 'ddr', capacity_bytes: raw.capacity_bytes};
assert.deepEqual(validatePhysicalMemoryConfig(raw, component), raw);
assert.equal(DRAM_BANK_GROUP_TIMING_FIELDS.length, 7);
const disabled = dramBankGroupTimingMarkup(raw);
assert.equal((disabled.match(/data-inspector-bank-group-timing=/g) || []).length, 7);
assert.ok(disabled.includes('全零关闭'));
assert.ok(disabled.includes('stack / die / lane / rank'));
assert.ok(disabled.includes('不填入任何厂商默认时序'));
for (const [field] of DRAM_BANK_GROUP_TIMING_FIELDS) {
  const active = {...raw, [field]: 7.25};
  assert.equal(validatePhysicalMemoryConfig(active, component)[field], 7.25);
  assert.ok(dramBankGroupTimingMarkup(active).includes('已启用'));
  const disabledOne = {...raw, ...Object.fromEntries(DRAM_BANK_GROUP_TIMING_FIELDS.map(([name]) => [name, 7.25])), [field]: 0};
  assert.deepEqual(validatePhysicalMemoryConfig(disabledOne, component), disabledOne);
  for (const value of [-1, Infinity, NaN, true, '1', null]) {
    assert.throws(() => validatePhysicalMemoryConfig({...raw, [field]: value}, component),
                  error => error.message.includes(field));
  }
}
assert.equal(dramBankGroupTimingMarkup({kind: 'HBF'}), '');
assert.equal(dramBankGroupTimingMarkup(null), '');
"""
    result = subprocess.run([node], input=script, text=True, encoding="utf-8",
                            capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
