import os
import sys
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
inner_dir = os.path.join(current_dir, "HiSTGNN")
if os.path.exists(inner_dir):
    os.chdir(inner_dir)

cmd = [sys.executable, "main.py"] + sys.argv[1:]
sys.exit(subprocess.call(cmd))
