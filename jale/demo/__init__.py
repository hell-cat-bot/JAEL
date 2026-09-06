"""Demo layer for JA-LE.

The V1 pipeline (``scripts/run_v1.py``) scores individual applications and audits
itself. This package adds the two layers the pitch describes but V1 did not
implement, plus the day-one workflow:

* ``l4_rings``   -- score the *ring*, not the application. Structural, no learned
                    weights, no labels.
* ``explain``    -- L5. Turn an application's score into a case note an
                    investigator can act on.
* ``coldstart``  -- propagate suspicion from a handful of confirmed cases with no
                    trained model at all.
* ``build_demo_data`` -- run everything once and emit a single JSON the demo UI
                    reads. No model runs in the browser.
"""
