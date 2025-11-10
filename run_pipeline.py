# run_pipeline.py - Complete Pipeline
import os
import sys
import subprocess
import webbrowser

def run_command(command, description):
    """Run a command and show output"""
    print(f"\n🎯 {description}")
    print("=" * 50)
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ SUCCESS")
            print(result.stdout)
        else:
            print("❌ ERROR")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    print("🐭 MOUSE BEHAVIOR - COMPLETE PIPELINE")
    print("=" * 60)
    
    # Step 1: Data preparation
    if not run_command("python src/data_loader.py", "Step 1: Data Loading Test"):
        return
    
    # Step 2: Training
    if not run_command("python src/train.py", "Step 2: Model Training"):
        return
    
    # Step 3: Submission generation
    if not run_command("python src/inference.py", "Step 3: Generate Submission"):
        return
    
    # Step 4: Check results
    if os.path.exists("submissions/competition_ready.csv"):
        print("\n🏆 PIPELINE COMPLETED SUCCESSFULLY!")
        print("📁 Generated files:")
        for file in os.listdir("submissions"):
            if file.endswith(".csv"):
                file_path = os.path.join("submissions", file)
                file_size = os.path.getsize(file_path)
                print(f"   ✅ {file} ({file_size} bytes)")
    else:
        print("\n❌ Pipeline failed - no submission file generated")

if __name__ == "__main__":
    main()