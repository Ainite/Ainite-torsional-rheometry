from pathlib import Path
import json
R=Path(__file__).resolve().parent/'results'
rule=json.load(open(R/'R182_PREREGISTERED_RULE.json'))
A=json.load(open(R/'case_edgewidth_0.05.json')); B=json.load(open(R/'case_edgewidth_0.15.json'))
valid=A['valid'] and B['valid']
if not valid: verdict='INVALID_NUMERICAL_GATE'
elif A['cancellation']['pass'] and A['approach']['pass'] and B['cancellation']['pass'] and B['approach']['pass']: verdict='TRANSITION_ONSET_TRANSLATION_SUPPORTED'
elif A['cancellation']['pass'] and B['cancellation']['pass']: verdict='RELATIVE_CANCELLATION_REPLICATED_APPROACH_SHAPE_MIXED'
else: verdict='TRANSITION_ONSET_TRANSLATION_NOT_SUPPORTED'
out={'schema':'U1-R182-transition-onset-translation-v2','preregistered_rule':rule,'cases':[A,B],'verdict':verdict,'posthoc_absolute_location_diagnostic':json.load(open(R/'R182_POSTHOC_ABSOLUTE_LOCATION.json')),'claim_boundary':'The preregistered relative-to-transition translation hypothesis failed for both altered transition widths. A post-hoc diagnostic shows strong cancellation remains near absolute 0.82-0.88, motivating the independently preregistered observer-window test R183; the post-hoc diagnostic does not alter the R182 verdict.'}
(R/'R182_TRANSITION_TRANSLATION_GATE.json').write_text(json.dumps(out,indent=2)+'\n'); print(verdict)
