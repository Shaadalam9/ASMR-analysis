## ASMR video analysis

## Getting started
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![Package Manager: uv](https://img.shields.io/badge/package%20manager-uv-green)](https://docs.astral.sh/uv/)

Tested with **Python 3.12** and the [`uv`](https://docs.astral.sh/uv/) package manager.  
Follow these steps to set up the project.

**Step 1:** Install `uv`. `uv` is a fast Python package and environment manager. Install it using one of the following methods:

**macOS / Linux (bash/zsh):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

**Alternative (if you already have Python and pip):**
```bash
pip install uv
```

**Step 2:** Fix permissions (if needed):t

Sometimes `uv` needs to create a folder under `~/.local/share/uv/python` (macOS/Linux) or `%LOCALAPPDATA%\uv\python` (Windows).  
If this folder was created by another tool (e.g. `sudo`), you may see an error like:
```lua
error: failed to create directory ... Permission denied (os error 13)
```

To fix it, ensure you own the directory:

### macOS / Linux
```bash
mkdir -p ~/.local/share/uv
chown -R "$(id -un)":"$(id -gn)" ~/.local/share/uv
chmod -R u+rwX ~/.local/share/uv
```

### Windows
```powershell
# Create directory if it doesn't exist
New-Item -ItemType Directory -Force "$env:LOCALAPPDATA\uv"

# Ensure you (the current user) own it
# (usually not needed, but if permissions are broken)
icacls "$env:LOCALAPPDATA\uv" /grant "$($env:UserName):(OI)(CI)F"
```

**Step 3:** After installing, verify:
```bash
uv --version
```

**Step 4:** Clone the repository:
```command line
git clone https://github.com/Shaadalam9/ASMR-analysis
cd multiped
```

**Step 5:** Ensure correct Python version. If you don’t already have Python 3.12 installed, let `uv` fetch it:
```command line
uv python install 3.12
```
The repo should contain a .python-version file so `uv` will automatically use this version.

**Step 6:** Create and sync the virtual environment. This will create **.venv** in the project folder and install dependencies exactly as locked in **uv.lock**:
```command line
uv sync --frozen
```

**Step 7:** Activate the virtual environment:

**macOS / Linux (bash/zsh):**
```bash
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (cmd.exe):**
```bat
.\.venv\Scripts\activate.bat
```

**Step 8:** Ensure that dataset are present. Place required datasets (including **mapping.csv**) into the **data/** directory:


**Step 9:** Run the code:
```command line
python3 analysis.py
```

### Configuration of project
Configuration of the project needs to be defined in `config`. Please use the `default.config` file for the required structure of the file. If no custom config file is provided, `default.config` is used. The config file has the following parameters:
- **`data`**: Path to the directory containing the project’s data files.
- **`query`**: The search query string used to discover videos (e.g., "ASMR", "ASMR roleplay", etc.).
- **`font_family`**: Specifies the font family to be used in outputs.
- **`font_size`**: Specifies the font size to be used in outputs.
- **`plotly_template`**: Defines the template for Plotly figures.
- **`logger_level`**: Level of console output. Can be: debug, info, warning, error.

### Results
### Word clouds

[![Word cloud of ASMR video titles](figures/wordcloud_titles.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/wordcloud_titles.html)  
Word cloud of ASMR video titles, generated from the collected dataset. More frequent terms appear larger, highlighting common themes and patterns in how creators title their ASMR videos.

[![Word cloud of ASMR video descriptions](figures/wordcloud_descriptions.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/wordcloud_descriptions.html)  
Word cloud of ASMR video descriptions, showing the most frequent terms used in the textual descriptions accompanying ASMR videos. This highlights how creators describe, contextualize, and promote their content.

[![Word cloud of ASMR video titles and descriptions](figures/wordcloud_titles_descriptions.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/wordcloud_titles_descriptions.html)  
Combined word cloud of ASMR video titles and descriptions, offering a holistic view of the most frequent terms across both fields. The visualization summarizes the overarching themes and stylistic patterns in the dataset.

---

### Duration vs popularity and engagement

[![Mean views by duration bucket](figures/duration_mean_views.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/duration_mean_views.html)  
Mean number of views per video, grouped by duration bucket (short, normal, long, other). This figure addresses whether longer ASMR sessions tend to attract more total views.

[![Mean views per day by duration bucket](figures/duration_mean_views_per_day.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/duration_mean_views_per_day.html)  
Mean views per day since upload, grouped by duration bucket. This captures *growth speed* and shows whether shorter or longer ASMR videos grow faster over time.

[![Mean engagement rate by duration bucket](figures/duration_mean_engagement_rate.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/duration_mean_engagement_rate.html)  
Mean engagement rate (likes / views) by duration bucket, exploring whether viewer engagement is higher for short videos, “normal” videos, or very long ASMR sessions.

---

### Language-level differences

[![Mean views per day by language](figures/language_mean_views_per_day.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/language_mean_views_per_day.html)  
Mean views per day by language (for languages with at least a minimum number of videos). This figure compares growth rates of ASMR content across languages (e.g., English vs Spanish vs others).

[![Mean engagement rate by language](figures/language_mean_engagement_rate.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/language_mean_engagement_rate.html)  
Mean engagement rate (likes / views) by language, showing which language communities tend to have more engaged ASMR audiences.

---

### Title style and performance

[![Mean engagement rate by title length](figures/title_length_mean_engagement_rate.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/title_length_mean_engagement_rate.html)  
Mean engagement rate by title length bucket (e.g., ≤5 words, 6–10 words, 11–20 words, >20 words). This figure examines whether concise or longer ASMR titles correlate with higher engagement.

[![Mean views by title length](figures/title_length_mean_views.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/title_length_mean_views.html)  
Mean total views by title length bucket, indicating whether shorter or more descriptive titles are associated with higher popularity.

---

### Themes vs growth (views per day distributions)

[![Views per day distribution for whisper videos](figures/has_whisper_views_per_day_boxplot.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/has_whisper_views_per_day_boxplot.html)  
Distribution of views per day for videos with and without “whisper” themes. This boxplot compares growth patterns between whisper-based ASMR and other content.

[![Views per day distribution for no-talking videos](figures/has_no_talking_views_per_day_boxplot.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/has_no_talking_views_per_day_boxplot.html)  
Distribution of views per day for videos tagged as “no talking” (or similar) versus others. This shows whether no-talking ASMR tends to grow faster or slower than talking-based ASMR.

[![Views per day distribution for sleep videos](figures/has_sleep_views_per_day_boxplot.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/has_sleep_views_per_day_boxplot.html)  
Distribution of views per day for sleep-oriented ASMR videos compared to non-sleep content, capturing whether “for sleep” videos exhibit different growth dynamics.

[![Views per day distribution for binaural videos](figures/has_binaural_views_per_day_boxplot.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/has_binaural_views_per_day_boxplot.html)  
Distribution of views per day for videos mentioning “binaural” or related keywords versus other videos, exploring whether binaural setups are associated with different growth rates.

---

### Community growth over time

[![Number of ASMR videos per month](figures/monthly_video_counts.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/monthly_video_counts.html)  
Time series of the number of ASMR videos uploaded per month in the dataset. This visualizes the growth of the ASMR ecosystem over time.

[![ASMR video uploads per year by language](figures/language_growth_over_years.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/language_growth_over_years.html)  
Yearly counts of ASMR uploads by language (for languages with enough data). This figure compares how quickly different language communities have expanded.

---

### Theme trends over time

[![Share of no-talking videos over time](figures/has_no_talking_trend_overall_fig.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/has_no_talking_trend_overall_fig.html)  
Share of videos tagged as “no talking” (or similar) over time, aggregated across all languages. This shows whether no-talking ASMR has become more or less prevalent.

[![Share of no-talking videos over time by language](figures/has_no_talking_trend_by_language_fig.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/has_no_talking_trend_by_language_fig.html)  
Share of “no talking” ASMR videos over time, broken down by language. This highlights cross-lingual differences in the adoption of no-talking formats.

[![Share of binaural videos over time](figures/has_binaural_trend_overall_fig.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/has_binaural_trend_overall_fig.html)  
Share of videos with “binaural” (and related) keywords over time, aggregated across languages. This plot reveals when binaural ASMR started to gain traction and how its prevalence has evolved.

[![Share of binaural videos over time by language](figures/has_binaural_trend_by_language_fig.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/has_binaural_trend_by_language_fig.html)  
Share of binaural ASMR videos over time, broken down by language, illustrating how different language communities adopted binaural production techniques.

---

### Seasonal patterns

[![Seasonal pattern of sleep videos](figures/sleep_seasonal_pattern_fig.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/sleep_seasonal_pattern_fig.html)  
Share of ASMR videos tagged as “sleep” across seasons (winter, spring, summer, autumn). This figure tests whether sleep-focused ASMR is more common in particular times of the year.

---

### Clustering of ASMR videos

[![Cluster sizes](figures/cluster_sizes.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/cluster_sizes.html)  
Number of videos per cluster, where clusters are derived from title+description text, duration, engagement metrics, and language. This shows how the ASMR corpus is partitioned into natural content groups.

[![Mean views per day by cluster](figures/cluster_mean_views_per_day.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/cluster_mean_views_per_day.html)  
Mean views per day for each cluster, comparing typical growth rates across the discovered ASMR content clusters.

[![2D embedding of ASMR video clusters](figures/cluster_scatter_embedding.png)](https://htmlpreview.github.io/?https://github.com/Shaadalam9/ASMR-analysis/blob/main/figures/cluster_scatter_embedding.html)  
2D embedding of ASMR videos based on title+description text, duration, engagement metrics, and language, with colors indicating clusters and dotted circles roughly outlining each cluster’s region. This offers an interpretable map of the ASMR content landscape.
