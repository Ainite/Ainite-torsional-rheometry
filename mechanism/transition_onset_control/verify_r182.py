from pathlib import Path
import json
R=Path(__file__).resolve().parent/'results'; d=json.load(open(R/'R182_TRANSITION_TRANSLATION_GATE.json'))
assert d['verdict']=='TRANSITION_ONSET_TRANSLATION_NOT_SUPPORTED'
assert all(c['valid'] for c in d['cases'])
assert all(not c['cancellation']['pass'] for c in d['cases'])
p=d['posthoc_absolute_location_diagnostic']['cases']
assert all(c['orth_angle_deg']>160 and c['reduction_fraction']>.75 for c in p)
print('R182 VERIFY PASS')
