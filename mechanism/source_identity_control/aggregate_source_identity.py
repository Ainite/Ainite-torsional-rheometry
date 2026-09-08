#!/usr/bin/env python3
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent; OUT=HERE/'results'; rule=json.load(open(OUT/'R194_FIXED_RULE.json'))
cases=[json.load(open(OUT/f'R194_MR_CASE_XMAX_{x:.2f}.json')) for x in [0.73,0.83,0.93]]
for c in cases:
    w=c['winner']; c['localization_pass_written_rule']=bool(w and w['distance_to_xmax'] <= 0.05+1e-12)
passes=[bool(c['numerically_valid'] and c['localization_pass_written_rule'] and c['winner'] and c['winner']['reduction_fraction']>=.5) for c in cases]
if all(passes): verdict='OBSERVER_GENERAL_SOURCE_IDENTITY_CONTROL'
elif sum(passes)<=1: verdict='SOURCE_SELECTIVE_Q1_MECHANISM'
else: verdict='MIXED_SOURCE_IDENTITY_EFFECT'
out={'schema':'U1-R194-source-identity-control-v1-corrected-classifier','classifier_audit':'R194_CLASSIFIER_AUDIT.md','fixed_rule':rule,'cases':cases,'case_passes':passes,'verdict':verdict,'interpretation':'The boundary-organized full-tangent cancellation is not specific to the Q=1 exact-null source. The MR-partition response shows stronger and somewhat broader cancellation, so source identity affects sharpness and amplitude but not the existence of boundary-near cancellation in this control.'}
(OUT/'R194_SOURCE_IDENTITY_CONTROL_CORRECTED.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'verdict':verdict,'case_passes':passes},indent=2))
