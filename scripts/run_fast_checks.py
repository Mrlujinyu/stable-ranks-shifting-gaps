"""Run regression and point-estimate checks; does not run full inference."""
import subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
for name in ['test_permutation_direction.py','test_identity_mapping.py','check_identity_full.py','audit_results.py']:
    subprocess.run([sys.executable,str(R/'scripts'/name)],cwd=R,check=True)
print('Fast checks complete. No bootstrap, new permutation, or GEE fit executed.')
