from pathlib import Path
import json
R=Path(__file__).resolve().parent/'results'; d=json.load(open(R/'R183_OBSERVER_WINDOW_TRANSLATION_GATE.json'))
assert d['verdict']=='OBSERVER_WINDOW_TRANSLATION_SUPPORTED'
assert d['valid'] and d['max_step_angle_deg']<.01
by={c['xmax']:c for c in d['cases']}
for x in [.75,.85,.95]:
    q=by[x]['cancellation']; assert q['pass'] and q['orth_angle_deg']>150 and q['reduction_fraction']>=.5
print('R183 VERIFY PASS')
