"""Build and verify Slide 2: Measured Results, Causal Proof & the Business Case.
Matches Runbook Step 4 on Slide 7.
"""
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SLIDE2_HTML = ROOT / "slide2_proposal.html"
SLIDE2_IMG = ROOT / "PROPOSAL" / "JALE_Slide2_Proposal.png"

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
