# by Shadab Alam <shaadalam.5u@gmail.com>

from logmod import logs
import common
from custom_logger import CustomLogger
from googleapiclient.discovery import build  # type: ignore
import os
import json
from types import SimpleNamespace


class ASMRFetcher:
    """Fetches ASMR videos from YouTube and stores new results.

    This class interacts with the YouTube Data API to search for ASMR videos,
    keeps track of previously seen video IDs, filters out YouTube Shorts using
    video duration, and appends new non-duplicate results to a JSON file.
    """

    def __init__(
        self,
        api_key,
        query="ASMR",
        max_pages=100,
        results_per_page=50,
        seen_file="seen_video_ids.txt",
        json_output="asmr_results_new.json",
    ):
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
        self.data_folder = "data"
        os.makedirs(self.data_folder, exist_ok=True)

        # Store files inside /data
        self.seen_file = os.path.join(self.data_folder, seen_file)
        self.json_output = os.path.join(self.data_folder, json_output)

        # Initialize logging
        logs(show_level=common.get_configs("logger_level"), show_color=True)
        self.logger = CustomLogger(__name__)

        # Initialize YouTube API client
        self.youtube = build("youtube", "v3", developerKey=self.api_key)

    def _is_short_video(self, iso_duration):
        """Determines whether a video is a YouTube Short based on ISO 8601 duration.

        YouTube Shorts are always **less than 60 seconds**.
        Duration formats examples:
            PT15S  -> 15 seconds
            PT45S  -> 45 seconds
            PT1M   -> 1 minute (NOT a short)
            PT2M10S -> 2 minutes 10 seconds

        Args:
            iso_duration (str): ISO 8601 duration string returned by API.

        Returns:
            bool: True if the video is considered a Short, otherwise False.
        """
        # If it contains minutes or hours, it's not a short
        if "M" in iso_duration or "H" in iso_duration:
            return False

        # If duration only has seconds → Shorts category
        # Example: PT30S
        return True

    def fetch_asmr_videos(self):
        """Fetches ASMR videos from YouTube, excluding Shorts, and stores results.

        Returns:
            list: Newly discovered ASMR videos (excluding Shorts).
        """
        # Load previously seen IDs
        if os.path.exists(self.seen_file):
            with open(self.seen_file, "r", encoding="utf-8") as f:
                seen_ids = set(line.strip() for line in f if line.strip())
        else:
            seen_ids = set()

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
                if video_id in seen_ids:
                    continue

                # ------------------------------------------------------------
                # EXCLUDE YOUTUBE SHORTS — Fetch duration
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

                # Mark video as seen
                seen_ids.add(video_id)

                # Store new valid ASMR video
                new_items.append({
                    "videoId": video_id,
                    "title": title,
                    "duration": iso_duration,
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
        # Merge old + new results without duplicates
        # ------------------------------------------------------------
        combined_by_id = {}

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

        # Update seen IDs file
        with open(self.seen_file, "w", encoding="utf-8") as f:
            for vid in seen_ids:
                f.write(vid + "\n")

        self.logger.info(
            f"Saved {len(new_items)} new entries to '{self.json_output}'. "
            f"Total entries stored: {len(combined_items)}"
        )
        self.logger.info(f"Total unique ASMR videos tracked: {len(seen_ids)}")

        return new_items


# ------------------------------------------------------------
# Instantiate fetcher and run
# ------------------------------------------------------------
secret = SimpleNamespace(
    API=common.get_secrets("google-api-key")
)

fetcher = ASMRFetcher(api_key=secret.API,
                      seen_file=os.path.join(common.get_configs("data"), "seen_video_ids.txt"),
                      json_output=os.path.join(common.get_configs("data"), "asmr_results.json"))
fetcher.fetch_asmr_videos()
