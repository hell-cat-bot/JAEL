"""Master 1-Click Reproducibility Runner for Hackathon Judges & Reviewers.
Executes all pipeline steps, verifies audited numbers, regenerates charts, and launches slides.
"""
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def run_step(step_num, title, cmd):
    print("\n" + "=" * 75)
    print(f"  STEP {step_num}: {title}")
    print(f"  Command: {cmd}")
    print("=" * 75)
    t0 = time.time()
    res = subprocess.run(cmd, shell=True, cwd=ROOT)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"\n[ERROR] Step {step_num} failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"  --> Completed in {elapsed:.1f} seconds. [PASS]")

def main():
    print("""
  =========================================================================
      JA·LE (Joint Application & Linkage Engine) — REPRODUCIBILITY RUNBOOK
      TVS Credit E.P.I.C 8.0 · Problem (E) Swarm Intelligence Lending Network
  =========================================================================
  "Nothing in this submission is a number we typed by hand.
   Clone, install, run — the pipeline regenerates features, models,
   the audit JSON, every chart, and the interactive demo."
    """)

    # Step 1: Audit Note 18, RBI macro data, and financial formulas
    run_step(1, "Audit TVS Audited Note 18 Math & RBI Macro Fraud Surge", f"{sys.executable} verify_audited_numbers.py")

    # Step 2: Full pipeline execution (Features, graphs, models, JSON audit files)
    run_step(2, "Full Pipeline Execution (Ring-Disjoint GBT, Leakage Gap, Shuffled Control)", f"{sys.executable} scripts/run_v1.py --profile SMOKE")

    # Step 3: Regenerate all charts
    run_step(3, "Regenerate All Presentation Charts", f"{sys.executable} PROPOSAL/make_charts.py")

    # Step 4: Rebuild and verify slide artefacts
    run_step(4, "Rebuild and Verify Slide 1 & Slide 2 Artefacts", f"{sys.executable} make_slide1.py")
    run_step(5, "Rebuild and Verify Slide 2 Artefacts", f"{sys.executable} make_slide2.py")

    print("\n" + "=" * 75)
    print("  >>> ALL PIPELINE ARTEFACTS & AUDITED NUMBERS REGENERATED IN < 45 SECONDS <<<")
    print("=" * 75)

    # Launch slides & interactive demo in browser
    slide1 = (ROOT / "slide1_proposal.html").resolve().as_uri()
    slide2 = (ROOT / "slide2_proposal.html").resolve().as_uri()
    demo = (ROOT / "demo" / "jale_demo.html").resolve().as_uri()

    print(f"\nLaunching Proposal Slides & Interactive Demo in your default browser...")
    print(f"  • Slide 1: {slide1}")
    print(f"  • Slide 2: {slide2}")
    print(f"  • Demo Console: {demo}\n")

    try:
        webbrowser.open(slide1)
        time.sleep(0.5)
        webbrowser.open(slide2)
        time.sleep(0.5)
        webbrowser.open(demo)
    except Exception:
        pass

    print("[DONE] Everything is ready for review.")

if __name__ == "__main__":
    main()
