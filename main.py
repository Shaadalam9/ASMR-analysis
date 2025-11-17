# by Shadab Alam <shaadalam.5u@gmail.com>

from logmod import logs
import common
from custom_logger import CustomLogger
from googleapiclient.discovery import build  # type: ignore
import os
import json
import re
from types import SimpleNamespace


class ASMRFetcher:
    """Fetches ASMR videos from YouTube and stores new results.

    This class interacts with the YouTube Data API to search for ASMR videos,
    keeps track of previously seen video IDs, filters out YouTube Shorts using
    video duration in seconds, and appends new non-duplicate results to a JSON file.
    """

    def __init__(self, api_key, query="ASMR", max_pages=100, results_per_page=50,
                 seen_file="seen_video_ids.txt", json_output="asmr_results_new.json"):
        """Initializes ASMRFetcher with API settings and storage paths.

        Args:
            api_key (str): YouTube API key.
            query (str, optional): Search query string. Defaults to "ASMR".
            max_pages (int, optional): Maximum number of API pages to fetch.
            results_per_page (int, optional): Number of items per API page.
            seen_file (str, optional): Filename for seen video IDs.
            json_output (str, optional): Filename for JSON results.
        """
        self.api_key = api_key
        self.query = query
        self.max_pages = max_pages
        self.results_per_page = results_per_page

        # Ensure data folder exists
        self.data_folder = common.get_configs("data")
        os.makedirs(self.data_folder, exist_ok=True)

        # Store files inside /data
        self.seen_file = os.path.join(self.data_folder, seen_file)
        self.json_output = os.path.join(self.data_folder, json_output)

        # Initialize logging
        logs(show_level=common.get_configs("logger_level"), show_color=True)
        self.logger = CustomLogger(__name__)

        # Initialize YouTube API client
        self.youtube = build("youtube", "v3", developerKey=self.api_key)

    # ------------------------------------------------------------
    # Duration helpers
    # ------------------------------------------------------------
    def _duration_to_seconds(self, iso_duration: str) -> int:
        """
        Convert YouTube ISO 8601 duration (e.g., 'PT1H2M10S', 'PT15S', 'PT1M', 'P0D')
        into total seconds.
        """

        # Special case: P0D -> 0 seconds
        if iso_duration == "P0D":
            return 0

        # YouTube usually returns 'PT...' formats
        yt_pattern = re.compile(
            r"^PT"
            r"(?:(\d+)H)?"
            r"(?:(\d+)M)?"
            r"(?:(\d+)S)?$"
        )
        match = yt_pattern.fullmatch(iso_duration)
        if match:
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            seconds = int(match.group(3) or 0)
            return hours * 3600 + minutes * 60 + seconds

        # More general ISO 8601 pattern: PnDTnHnMnS, with optional T-part
        generic_pattern = re.compile(
            r"^P"
            r"(?:(\d+)D)?"
            r"(?:T"
            r"(?:(\d+)H)?"
            r"(?:(\d+)M)?"
            r"(?:(\d+)S)?"
            r")?$"
        )
        match = generic_pattern.fullmatch(iso_duration)
        if match:
            days = int(match.group(1) or 0)
            hours = int(match.group(2) or 0)
            minutes = int(match.group(3) or 0)
            seconds = int(match.group(4) or 0)
            return (((days * 24) + hours) * 60 + minutes) * 60 + seconds

        # Fallback: unknown format
        self.logger.warning(f"Could not parse duration: {iso_duration}")
        return 0

    def _is_short_video(self, iso_duration: str) -> bool:
        """Determines whether a video is a YouTube Short (< 60 seconds)."""
        total_seconds = self._duration_to_seconds(iso_duration)
        return total_seconds < 60

    # ------------------------------------------------------------
    # Main fetch logic
    # ------------------------------------------------------------
    def fetch_asmr_videos(self):
        """Fetches ASMR videos from YouTube, excluding Shorts, and stores results.

        Returns:
            list: Newly discovered ASMR videos (excluding Shorts).
        """
        # ------------------------------------------------------------
        # Load previously seen IDs in a stable (ordered) way
        # ------------------------------------------------------------
        seen_ids_list = []  # preserves insertion order as in file
        seen_ids_set = set()  # fast membership test

        if os.path.exists(self.seen_file):
            with open(self.seen_file, "r", encoding="utf-8") as f:
                for line in f:
                    vid = line.strip()
                    if vid and vid not in seen_ids_set:
                        seen_ids_list.append(vid)
                        seen_ids_set.add(vid)

        new_items = []
        next_page_token = None

        # ------------------------------------------------------------
        # Fetch pages of YouTube search results
        # ------------------------------------------------------------
        for _ in range(self.max_pages):
            request = self.youtube.search().list(
                q=self.query,
                part="snippet",
                type="video",
                maxResults=self.results_per_page,
                pageToken=next_page_token,
                order="date",
            )
            response = request.execute()

            for item in response["items"]:
                video_id = item["id"]["videoId"]
                title = item["snippet"]["title"]

                # Skip if already processed previously
                if video_id in seen_ids_set:
                    continue

                # ------------------------------------------------------------
                # Fetch duration and exclude Shorts
                # ------------------------------------------------------------
                details = self.youtube.videos().list(
                    part="contentDetails",
                    id=video_id
                ).execute()

                # Some items can fail to fetch details; skip safely
                if not details.get("items"):
                    continue

                iso_duration = details["items"][0]["contentDetails"]["duration"]

                # Skip Shorts based on duration (<60 seconds)
                if self._is_short_video(iso_duration):
                    continue

                duration_seconds = self._duration_to_seconds(iso_duration)

                # Mark video as seen (append at end to keep file stable)
                seen_ids_list.append(video_id)
                seen_ids_set.add(video_id)

                # Store new valid ASMR video
                new_items.append({
                    "videoId": video_id,
                    "title": title,
                    "duration": duration_seconds,  # store as seconds
                    # Optional: keep original string too:
                    # "duration_iso": iso_duration,
                })

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        self.logger.info(f"Found {len(new_items)} new non-short ASMR videos.")

        # ------------------------------------------------------------
        # Load existing JSON data
        # ------------------------------------------------------------
        existing_items = []
        if os.path.exists(self.json_output):
            try:
                with open(self.json_output, "r", encoding="utf-8") as f:
                    existing_items = json.load(f)
            except (json.JSONDecodeError, ValueError, TypeError):
                existing_items = []

        # ------------------------------------------------------------
        # Merge old + new results without duplicates (by videoId)
        # ------------------------------------------------------------
        combined_by_id = {}

        # Existing first, then new overwrites if same ID
        for item in existing_items:
            vid = item.get("videoId")
            if vid:
                combined_by_id[vid] = item

        for item in new_items:
            vid = item.get("videoId")
            if vid:
                combined_by_id[vid] = item

        combined_items = list(combined_by_id.values())

        # Save merged results to JSON
        with open(self.json_output, "w", encoding="utf-8") as f:
            json.dump(combined_items, f, indent=4, ensure_ascii=False)

        # ------------------------------------------------------------
        # Update seen IDs file in a STABLE order
        # (existing order preserved, new IDs appended at end)
        # ------------------------------------------------------------
        with open(self.seen_file, "w", encoding="utf-8") as f:
            for vid in seen_ids_list:
                f.write(vid + "\n")

        self.logger.info(
            f"Saved {len(new_items)} new entries to '{self.json_output}'. "
            f"Total entries stored: {len(combined_items)}"
        )
        self.logger.info(f"Total unique ASMR videos tracked: {len(seen_ids_list)}")

        return new_items


# ------------------------------------------------------------
# Instantiate fetcher and run
# ------------------------------------------------------------
secret = SimpleNamespace(
    API=common.get_secrets("google-api-key")
)

fetcher = ASMRFetcher(
    api_key=secret.API,
    query="ASMR",
    max_pages=100,                # increase if you want up to more pages
    results_per_page=50,          # YouTube API max
    seen_file="seen_video_ids.txt",
    json_output="asmr_results.json"
)

if __name__ == "__main__":
    fetcher.fetch_asmr_videos()
