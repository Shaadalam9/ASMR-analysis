Supplementary material for the data article: {citation}


It has the following structure:
* data/
*     data/asmr_videos.csv.gz: the video table, 90,189 rows and 45 variables (gzip-compressed CSV).
*     data/asmr_results.json.gz: the complete source snapshot exactly as collected (gzip-compressed JSON).
*     data/seen_video_ids.txt: the 90,189 video identifiers seen by the collector, one per line.
*     data/data_dictionary.csv: name, type, unit, description and missing count of each variable of the table.
*     data/raw_snapshot_schema.csv: fields of the source snapshot.
*     data/theme_rules.csv: the ten regular expressions used for the theme indicators, with the rule version.
*     data/reproducibility_manifest.json: SHA-256 digest of the source snapshot, record count, reference date, rule version, random seed, and Python and package versions.
*     data/release_summary.json: number of videos, channels and language categories, date range and file sizes.
*     data/SHA256SUMS.txt: SHA-256 checksums of the other files in data/.
* code/
*     code/.python-version: Specifies the Python version used for the project.
*     code/analysis.py: Entry point of the analysis; runs asmr/analysis/publication.py, which produces the derived measures, tables, figures, clusters and UMAP.
*     code/main.py: Entry point of the video discovery (collection) step; not needed to reproduce the analysis.
*     code/asmr/: Python package with the collection, processing, analysis and visualisation code.
*     code/asmr/analysis/supplementary/: Supplementary analyses (theme-rule audit, diversification and concentration, models, robustness, cluster diagnostics).
*     code/asmr/analysis/release.py: Builds the files in data/ and this package from the analysis outputs.
*     code/tests/: Deterministic unit tests that do not require access to YouTube.
*     code/common.py: Contains functions for configuration management, secrets, dictionary search, and data serialisation.
*     code/custom_logger.py: Implements a custom logger class for handling string formatting and logging at various levels.
*     code/logmod.py: Initialises and configures the logger with customisable display and storage options.
*     code/default.config: Configuration file specifying the data location and analysis settings.
*     code/default.secret: Template of the secret file with the YouTube API keys (needed only for collection).
*     code/pyproject.toml and code/uv.lock: Dependencies and their locked versions.
*     code/LICENSE and code/SECURITY.md: Licence and security policy.


To run analysis code, do the following replace "/" with "\" and possibly add "--user" if on Windows:
1. Install uv using "curl -LsSf https://astral.sh/uv/install.sh | sh" on macOS/Linux, or "irm https://astral.sh/uv/install.ps1 | iex" on Windows PowerShell. Alternatively, use "pip install uv" or "pip install uv --user" if needed.
2. If uv gives a permission error for "~/.local/share/uv/python" on macOS/Linux or "%LOCALAPPDATA%\uv\python" on Windows, fix permissions. On macOS/Linux, run "mkdir -p ~/.local/share/uv", then "chown -R "$(id -un)":"$(id -gn)" ~/.local/share/uv", then "chmod -R u+rwX ~/.local/share/uv".
3. Verify uv is installed with "uv --version".
4. Verify the files in the folder "data" with "cd data" and "shasum -a 256 -c SHA256SUMS.txt" (on Windows PowerShell use "Get-FileHash" and compare with SHA256SUMS.txt), then decompress the source snapshot with "gunzip -k asmr_results.json.gz". The analysis reads "data/asmr_results.json".
5. Move to the folder with the analysis code with the command "cd ../code".
6. Create the configuration file with "cp default.config config" (on Windows PowerShell "Copy-Item default.config config"). The code stops with an error if the file "config" is missing. The default values reproduce the reported analysis; "data" already points to "../data".
7. Ensure Python 3.12 is available. If needed, install it with "uv python install 3.12". The folder contains a ".python-version" file, so uv will automatically use this version.
8. Create the virtual environment and install dependencies with "uv sync --frozen". This creates ".venv" and installs packages from "uv.lock".
9. Activate the virtual environment with "source .venv/bin/activate" on macOS/Linux, ".venv\Scripts\Activate.ps1" on Windows PowerShell, or ".venv\Scripts\activate.bat" on Windows cmd.exe.
10. Run the analysis code with "python3 analysis.py" on macOS/Linux or "python analysis.py" on Windows. Outputs are written to "_output/" and figures to "figures/".
11. Optionally run the supplementary analyses with "python3 -m asmr.analysis.supplementary" and the unit tests with "python3 -m unittest discover -s tests".


Reading the data:
In Python, read the table with pd.read_csv("asmr_videos.csv.gz", keep_default_na=False, na_values=[""], low_memory=False). The keep_default_na=False argument keeps text such as "NA" as text. Missing values are empty cells.


Configuration of project:
Configuration of the project needs to be defined in "code/config". Please see the "code/default.config" file for the required structure of the file. Create it by copying "code/default.config" (step 6). The config file needs to have the following parameters:
{config_lines}


The "secret" file:
API keys are stored in a file named "secret" (no extension) in the "code" folder. It is needed only for collecting or refreshing data, not for the analysis. Copy "code/default.secret" to "code/secret" and fill in the value. The file is one JSON object with the key "google-api-keys": a list of YouTube Data API v3 keys, which are rotated when the daily quota of one key is used up. Do not share the file.


Notes:
* The code was tested with Python 3.12.13.
* The data are public video metadata obtained through the YouTube Data API and the public search interface. Please comply with the YouTube Terms of Service and applicable data protection law.
* The snapshot does not store per-record collection timestamps; counts were retrieved at different times.
* The dataset is also published and openly available in Zenodo: https://doi.org/10.5281/zenodo.23574347 (DOI 10.5281/zenodo.23574347). The code is archived in Zenodo: https://doi.org/10.5281/zenodo.23398907.
