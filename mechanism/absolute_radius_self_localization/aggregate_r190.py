#!/usr/bin/env python3
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent; OUT=HERE/'results'
rule=json.loads((OUT/'R190_PREREGISTERED_RULE.json').read_text())
cases=[json.loads((OUT/f'R190_CASE_XMAX_{x:.2f}.json').read_text()) for x in rule['new_observer_xmax']]
for c in cases:
    numerical_valid=bool(c['max_step_angle_deg']<.01 and c['baseline_grad_inf_pre']<1e-7 and c['baseline_grad_inf_twist']<1e-7)
    mechanics_pass=bool(c['winner_mechanics'] is not None and c['winner_mechanics']['pass'])
    localization_pass=bool(c['localization_pass'])
    c['numerical_valid']=numerical_valid
    c['mechanics_pass']=mechanics_pass
    c['pass_corrected_classifier']=bool(numerical_valid and mechanics_pass and localization_pass)
if all(c['pass_corrected_classifier'] for c in cases):
    verdict='PROSPECTIVE_OBSERVER_SELF_LOCALIZATION_SUPPORTED'
elif any(not c['numerical_valid'] for c in cases):
    verdict='INVALID_NUMERICAL_GATE'
else:
    verdict='PROSPECTIVE_OBSERVER_SELF_LOCALIZATION_NOT_SUPPORTED'
out={
  'schema':'U1-R190-prospective-self-localization-v1-classifier-corrected',
  'preregistered_rule':rule,
  'classifier_correction':{
    'reason':'Original implementation conflated winner_mechanics failure with numerical validity in case valid flag. This correction separates the frozen numerical_validity clause from the frozen winner_mechanics_gate clause; no threshold, candidate, winner, or scientific number was changed.',
    'raw_aggregate':'R190_PROSPECTIVE_SELF_LOCALIZATION_GATE_RAW_CLASSIFIER.json'
  },
  'cases':cases,
  'verdict':verdict
}
(OUT/'R190_PROSPECTIVE_SELF_LOCALIZATION_GATE.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'verdict':verdict,'summary':[{'xmax':c['xmax'],'winner':c['winner']['midpoint'] if c['winner'] else None,'distance':c['winner']['distance_to_xmax'] if c['winner'] else None,'reduction':c['winner']['reduction_fraction'] if c['winner'] else None,'numerical_valid':c['numerical_valid'],'mechanics_pass':c['mechanics_pass'],'pass':c['pass_corrected_classifier']} for c in cases]},indent=2))
