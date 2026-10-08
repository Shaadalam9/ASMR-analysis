"""Entry point for video discovery (used by the cron job): ``python main.py``."""
import runpy

if __name__ == "__main__":
    runpy.run_module("asmr.collection.discover", run_name="__main__")
