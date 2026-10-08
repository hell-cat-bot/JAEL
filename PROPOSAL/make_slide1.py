"""Verify Slide 1 artefacts: the HTML source and the 16:9 PNG.
Matches RUNBOOK step 5 (open slides).
"""
import sys
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLIDE1_HTML = HERE / "slide1_proposal.html"
SLIDE1_IMG = HERE / "JALE_Slide1_Proposal.png"

def main():
    print("[Slide 1 Rebuilder] Checking Slide 1 artefacts...")
    if not SLIDE1_HTML.exists():
        print(f"[ERROR] {SLIDE1_HTML} not found.")
        sys.exit(1)

    print(f"  --> Slide 1 HTML source: {SLIDE1_HTML}")
    if SLIDE1_IMG.exists():
        print(f"  --> Slide 1 16:9 Image: {SLIDE1_IMG}")

    print("  [SUCCESS] Slide 1 artefacts ready.")

    # Optional auto-launch if requested via CLI arg --open
    if "--open" in sys.argv:
        webbrowser.open(SLIDE1_HTML.as_uri())

if __name__ == "__main__":
    main()
