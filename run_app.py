"""RipeSense — single entry point.

Run this file to start the decision-support dashboard:

    py run_app.py

It checks the Python version, installs any missing dependencies, makes sure the
trained models are present (running the experiment pipeline once if they are
not), and then opens the Streamlit app in the default browser.

To reproduce the experiments instead of launching the dashboard, run:

    py -m src.run_pipeline
"""

import importlib.util
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "outputs", "models")

#: Import name -> pip requirement, for the packages the dashboard cannot start
#: without.
CORE_PACKAGES = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "joblib": "joblib",
    "plotly": "plotly",
    "networkx": "networkx",
    "shap": "shap",
}

REQUIRED_ARTEFACTS = ("scaler.pkl", "kg_generator.json", "baseline_rf.pkl",
                      "kg_rf.pkl")


def say(message=""):
    # Unbuffered so progress is visible even when output is piped to a log.
    print(message, flush=True)


def check_python():
    if sys.version_info < (3, 11):
        say(f"Python 3.11 or newer is required (found {sys.version.split()[0]}).")
        sys.exit(1)
    say(f"Python {sys.version.split()[0]} - OK")


def install_dependencies():
    missing = [pkg for mod, pkg in CORE_PACKAGES.items()
               if importlib.util.find_spec(mod) is None]
    if not missing:
        say("All dependencies already installed - OK")
        return
    say("Installing dependencies: " + ", ".join(missing))
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r",
         os.path.join(ROOT, "requirements.txt")],
        cwd=ROOT)
    if result.returncode != 0:
        say("\nDependency installation failed. Install them manually with:")
        say("    py -m pip install -r requirements.txt")
        sys.exit(result.returncode)


def ensure_models():
    absent = [f for f in REQUIRED_ARTEFACTS
              if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if not absent:
        say("Trained models found - OK")
        return
    say("Trained models missing: " + ", ".join(absent))
    say("Running the experiment pipeline once (about 4-5 minutes)...")
    result = subprocess.run([sys.executable, "-m", "src.run_pipeline"], cwd=ROOT)
    if result.returncode != 0:
        say("\nThe pipeline did not finish. Run it manually to see the error:")
        say("    py -m src.run_pipeline")
        sys.exit(result.returncode)


def launch_app():
    say("\nStarting the dashboard at http://localhost:8501")
    say("Press Ctrl+C in this window to stop it.\n")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run",
                        os.path.join(ROOT, "app.py")], cwd=ROOT)
    except KeyboardInterrupt:
        say("\nDashboard stopped.")


def main():
    say("=" * 62)
    say("RipeSense - Knowledge-Integrated Banana Ripeness Prediction")
    say("=" * 62)
    check_python()
    install_dependencies()
    ensure_models()
    launch_app()


if __name__ == "__main__":
    main()
