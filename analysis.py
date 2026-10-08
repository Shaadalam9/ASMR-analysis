"""Entry point for the publication analysis: ``python analysis.py``."""
import runpy

if __name__ == "__main__":
    runpy.run_module("asmr.analysis.publication", run_name="__main__")
