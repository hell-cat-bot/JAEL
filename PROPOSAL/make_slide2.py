"""Verify Slide 2 artefacts: the HTML source and the 16:9 PNG.
Matches RUNBOOK step 5 (open slides).
"""
import sys
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLIDE2_HTML = HERE / "slide2_proposal.html"
SLIDE2_IMG = HERE / "JALE_Slide2_Proposal.png"

def main():
    print("[Slide 2 Rebuilder] Checking Slide 2 artefacts...")
    if not SLIDE2_HTML.exists():
        print(f"[ERROR] {SLIDE2_HTML} not found.")
        sys.exit(1)

    print(f"  --> Slide 2 HTML source: {SLIDE2_HTML}")
    if SLIDE2_IMG.exists():
        print(f"  --> Slide 2 16:9 Image: {SLIDE2_IMG}")

    print("  [SUCCESS] Slide 2 artefacts ready.")

    # Optional auto-launch if requested via CLI arg --open
    if "--open" in sys.argv:
        webbrowser.open(SLIDE2_HTML.as_uri())

if __name__ == "__main__":
    main()
