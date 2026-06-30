#!/usr/bin/env bash
set -e

git add -A                                   # stage everything outstanding
# Edit message as needed
git commit -m "Remove .gitignore"
#git commit -m "Code review: fix docstrings, remove stale API refs, spelling, minor guards
#
#- Fix ExponentialAnneal docstring example (schedule(20) was wrong)
#- Remove target_epochs attribute (was computed from wrong formula)
#- Add defensive else branch to Embedding._compute_dist
#- Narrow ImportError catch in _snapshot_dr_metrics to pyDRMetrics only
#- Clarify MQE is RMSE throughout (vqlp QE stores squared L2 distance)
#- Add 2-D shape guard on X in from_grid
#- Add explicit n_epochs param to fit_transform with updated docstring
#- Change plot() default title from 'SOUMAP' to 'GTSOM'
#- Add __all__ to vis_tools.py; export theme_minimal_bold from package
#- Remove all references to defunct compile() method"
git push
mkdocs gh-deploy
