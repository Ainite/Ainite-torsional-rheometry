#!/usr/bin/env python3
from pathlib import Path
import json, hashlib, sys
HERE=Path(__file__).resolve().parent
pre=HERE/'R208_PREREGISTRATION.json'; result=HERE/'results/R208_THIN_GAP_RELEVANCE_RESULT.json'
h=hashlib.sha256(pre.read_bytes()).hexdigest(); d=json.loads(result.read_text()); rule=json.loads(pre.read_text())
checks={
 'prereg_hash': d['preregistration_sha256']==h,
 'numerical_validity': bool(d['pass_flags']['numerical_validity']),
 'finite_response': d['constitutive_tangent']['published_amplitude_response_SD']>=rule['pass_rules']['finite_response_SD_min'],
 'interface_overlap': d['constitutive_tangent']['outside_interface_span_fraction']<=rule['pass_rules']['outside_interface_span_fraction_max'],
 'verdict': d['verdict']=='THIN_GAP_RELEVANCE_SUPPORTED'
}
print(json.dumps({'checks':checks,'pass':all(checks.values())},indent=2))
sys.exit(0 if all(checks.values()) else 1)
