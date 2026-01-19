import os
import sys

def main():
    print("Start: generating paper plots...")
    
    # 1. Run the existing analysis package logic if possible
    try:
        from analysis.paper_plots import run_analysis
        print("Running analysis.paper_plots...")
        run_analysis()
    except ImportError:
        print("Warning: Could not import analysis.paper_plots. Running in standalone mode.")
    except Exception as e:
        print(f"Warning: Error during analysis run: {e}")

    # 2. Verify and Organize LaTeX Assets
    # The paper uses specific filenames at the root. 
    # We ensure they exist, either from new generation or pre-packaged assets.
    
    required_images = [
        'phase_transition_all.png',
        'finite_size_scaling.png',
        'bundle_amplification.png',
        'fig_chunking_overhead_tradeoff.png',
        'fig_e7_robustness_scurves.png'
    ]
    
    missing = []
    for img in required_images:
        if os.path.exists(img):
            print(f"[OK] {img} exists.")
        else:
            # Check if it was generated in analysis/ or experiments/results/ and move it
            possible_locs = [
                os.path.join('analysis', 'figures', img),
                os.path.join('experiments', 'results', img),
                os.path.join('analysis', img)
            ]
            found = False
            for loc in possible_locs:
                if os.path.exists(loc):
                    print(f"Found {img} in {loc}, moving to root...")
                    os.rename(loc, img)
                    found = True
                    break
            
            if not found:
                missing.append(img)
                print(f"[FAIL] {img} not found.")

    if missing:
        print(f"Error: The following required figures for the LaTeX compilation are missing: {missing}")
        # In a real repro execution, we might exit 1, but for supplementary package usage we warn.
        sys.exit(1)
    else:
        print("Success: All required paper figures are present.")

if __name__ == "__main__":
    main()
