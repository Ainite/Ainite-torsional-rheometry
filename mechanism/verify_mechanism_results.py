#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
EXPECTED={
 'cross_discretization_geometry/results/R181_CROSS_REPLICATION_GATE.json':'CANCELLATION_REPLICATED_LOCATION_NOT_REPLICATED',
 'transition_onset_control/results/R182_TRANSITION_TRANSLATION_GATE.json':'TRANSITION_ONSET_TRANSLATION_NOT_SUPPORTED',
 'observer_window_translation/results/R183_OBSERVER_WINDOW_TRANSLATION_GATE.json':'OBSERVER_WINDOW_TRANSLATION_SUPPORTED',
 'nested_direct_state/results/R184_NESTED_CANCELLATION_GATE.json':'NESTED_OBSERVER_BOUNDARY_CANCELLATION_SUPPORTED',
 'loading_amplitude/results/R185_LOADING_AMPLITUDE_GATE.json':'LOADING_AMPLITUDE_NESTED_CANCELLATION_REPLICATED',
 'material_baseline/results/R186_MATERIAL_BASELINE_GATE.json':'MATERIAL_BASELINE_NESTED_CANCELLATION_REPLICATED',
 'absolute_radius_self_localization/results/R190_PROSPECTIVE_SELF_LOCALIZATION_GATE.json':'PROSPECTIVE_OBSERVER_SELF_LOCALIZATION_NOT_SUPPORTED',
 'boundary_neighborhood_nested/results/R192_BOUNDARY_NEIGHBORHOOD_GATE.json':'PROSPECTIVE_BOUNDARY_NEIGHBORHOOD_NESTED_CANCELLATION_SUPPORTED',
}
for rel,expected in EXPECTED.items():
 p=ROOT/rel; data=json.loads(p.read_text(encoding='utf-8')); got=data.get('verdict')
 if got!=expected: raise SystemExit(f'FAIL: {rel}: expected {expected!r}, got {got!r}')
 print(f'PASS: {rel}: {got}')
# Extra semantic checks for hardening controls.
r190=json.loads((ROOT/'absolute_radius_self_localization/results/R190_PROSPECTIVE_SELF_LOCALIZATION_GATE.json').read_text())
assert all(c['localization_pass'] for c in r190['cases'])
assert all(c['numerical_valid'] for c in r190['cases'])
assert sum(bool(c['mechanics_pass']) for c in r190['cases'])==1
r192=json.loads((ROOT/'boundary_neighborhood_nested/results/R192_BOUNDARY_NEIGHBORHOOD_GATE.json').read_text())
assert all(c['pass'] for c in r192['cases'])
assert all(all(b['band_pass'] for b in c['bands']) for c in r192['cases'])
print('MECHANISM VERIFICATION PASS')
