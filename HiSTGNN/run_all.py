import subprocess
import sys
import os
import argparse

def run_experiment(dataset_name, device, epochs):
    print("\n" + "="*70)
    print(f">> STARTING BENCHMARK FOR: {dataset_name.upper()}")
    print("="*70 + "\n")
    
    cmd = [
        sys.executable, "main.py",
        "--data_name", dataset_name,
        "--device", device,
        "--epochs", str(epochs)
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[FAIL] Execution failed for {dataset_name} with exit code {result.returncode}")
        return False
    else:
        print(f"\n[SUCCESS] Completed benchmark for {dataset_name}")
        return True

def main():
    parser = argparse.ArgumentParser(description="Run HiSTGNN across all benchmark datasets")
    parser.add_argument('--device', type=str, default='auto', help='cuda:0, cpu, or auto')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs per dataset (default: 100)')
    parser.add_argument('--datasets', nargs='+', default=['BJ', 'Israel', 'USA'],
                        choices=['BJ', 'Israel', 'USA'], help='Datasets to run')
    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        try:
            import torch
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'
    else:
        device = args.device

    print("="*70)
    print("HiSTGNN UNIFIED MULTI-DATASET RUNNER")
    print(f"Target Datasets: {', '.join(args.datasets)}")
    print(f"Device: {device}")
    print(f"Epochs per dataset: {args.epochs}")
    print("="*70)

    results = {}
    for ds in args.datasets:
        success = run_experiment(ds, device, args.epochs)
        results[ds] = "PASSED" if success else "FAILED"

    print("\n" + "="*70)
    print("FINAL SUMMARY OF RUNS:")
    for ds, status in results.items():
        print(f"  - {ds}: {status}")
    print("="*70)

if __name__ == '__main__':
    main()
