# by Shadab Alam <shaadalam.5u@gmail.com>

from logmod import logs
import common
from custom_logger import CustomLogger
from googleapiclient.discovery import build  # type: ignore
from pytubefix import YouTube
from pytubefix.contrib.search import Search, Filter
import os
import json
import re
from types import SimpleNamespace
from typing import Dict, Any, List, Optional


class ASMRFetcher:
    """Fetch, enrich, and persist ASMR videos from YouTube.

    This class discovers ASMR videos through two mechanisms:

    1. YouTube Data API (if an API key is provided).
    2. pytubefix contrib Search (web scraping; no API key required).

    It then:
      * Filters out short videos (YouTube Shorts) based on duration (< 60 seconds).
      * Enriches videos using pytubefix with:
          - title
          - duration
          - channelId
          - author
          - views
          - likes
          - description
      * Ensures ASMR relevance with a case-insensitive "asmr" check in title/description.
      * Deduplicates videos by `videoId`.
      * Stores all results in a JSON file with the structure:

        {
            "<videoId>": {
                "title": "...",
                "duration": 1234,
                "channelId": "...",
                "author": "...",
                "views": 12345,
                "likes": 678,
                "description": "..."
            },
            ...
        }

    The deduplication is based on both the JSON contents and a separate
    `seen_video_ids.txt` file, so repeated runs do not re-add existing videos.

    Attributes:
        api_key (Optional[str]): YouTube Data API key, or None if not provided.
        query (str): Search query used for ASMR discovery.
        max_pages (int): Max pages to fetch in the YouTube API and as a soft cap
            for pytubefix Search.
        results_per_page (int): Results per page in the YouTube API.
        data_folder (str): Directory path to hold JSON and seen IDs file.
        seen_file (str): Path to the file storing already seen video IDs.
        json_output (str): Path to the JSON file storing video metadata.
        youtube: YouTube Data API client or None if no API key is set.
        enforce_asmr_keyword (bool): If True, only keep videos where
            title/description contain "asmr" (case-insensitive).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        query: str = "ASMR",
        max_pages: int = 100,
        results_per_page: int = 50,
        seen_file: str = "seen_video_ids.txt",
        json_output: str = "asmr_results_new.json",
    ) -> None:
        """Initialize ASMRFetcher.

        Args:
            api_key (Optional[str]): YouTube API key. If None or empty, only
                pytubefix contrib Search is used for discovery.
            query (str): Search query string used for discovery. Defaults to "ASMR".
            max_pages (int): Maximum number of pages to fetch in YouTube Data API
                results and also used as a soft cap for pytubefix Search.
            results_per_page (int): Number of search results per page in the
                YouTube Data API. YouTube usually caps this at 50.
            seen_file (str): File name (inside the configured data folder) storing
                previously seen video IDs, one per line.
            json_output (str): File name (inside the configured data folder) storing
                the JSON results.

        Raises:
            KeyError: If `common.get_configs("data")` is missing or misconfigured.
        """
        self.api_key = api_key or None
        self.query = query
        self.max_pages = max_pages
        self.results_per_page = results_per_page

        # Ensure data folder exists
        self.data_folder = common.get_configs("data")
        os.makedirs(self.data_folder, exist_ok=True)

        # Store files inside configured /data folder
        self.seen_file = os.path.join(self.data_folder, seen_file)
        self.json_output = os.path.join(self.data_folder, json_output)

        # Initialize logging
        logs(show_level=common.get_configs("logger_level"), show_color=True)
        self.logger = CustomLogger(__name__)

        # Initialize YouTube API client only if API key is given
        if self.api_key:
            self.youtube = build("youtube", "v3", developerKey=self.api_key)
            self.logger.info("YouTube Data API enabled (API key provided).")
        else:
            self.youtube = None
            self.logger.info("No API key provided. Using only pytubefix Search for discovery.")

        # If the query string contains "asmr" (case-insensitive),
        # enforce the keyword in title/description later.
        self.enforce_asmr_keyword = "asmr" in (self.query or "").lower()

    # -------------------------------------------------------------------------
    # Duration helpers
    # -------------------------------------------------------------------------
    def _duration_to_seconds(self, iso_duration: str) -> int:
        """Convert an ISO 8601 YouTube duration string to total seconds.

        This helper interprets common YouTube duration formats such as:
          * "PT1H2M10S"
          * "PT15S"
          * "PT1M"
          * "P0D" (special case for zero seconds)

        Args:
            iso_duration (str): ISO 8601 duration string as returned by
                the YouTube Data API in video `contentDetails.duration`.

        Returns:
            int: Total number of seconds represented by the duration. If the
            duration cannot be parsed, 0 is returned and a warning is logged.
        """
        if iso_duration == "P0D":
            return 0

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

        self.logger.warning(f"Could not parse duration: {iso_duration}")
        return 0

    def _is_short_video(self, seconds: int) -> bool:
        """Check whether a video should be considered a YouTube Short.

        Args:
            seconds (int): Duration of the video in seconds.

        Returns:
            bool: True if `seconds` is strictly less than 60, indicating a short video.
        """
        return seconds < 60

    # -------------------------------------------------------------------------
    # pytubefix metadata helpers
    # -------------------------------------------------------------------------
    def _normalize_int(self, val: Any) -> Any:
        """Normalize numeric fields (views/likes) to integers when possible.

        Args:
            val (Any): Value to normalize, possibly int, float, or string.

        Returns:
            Any: Integer if the value can be safely converted; otherwise the
            original value.
        """
        if isinstance(val, int):
            return val
        if isinstance(val, float):
            return int(val)
        if isinstance(val, str):
            s = val.replace(",", "").strip()
            return int(s) if s.isdigit() else val
        return val

    def _safe_int_or_zero(self, val: Any) -> int:
        """Convert a value to int if reasonable, otherwise return 0.

        This is used for duration-like fields where failure should not raise.

        Args:
            val (Any): Value that may represent an integer.

        Returns:
            int: Parsed integer value, or 0 if parsing is not possible.
        """
        if isinstance(val, int):
            return val
        if isinstance(val, float):
            return int(val)
        if isinstance(val, str):
            s = val.replace(",", "").strip()
            return int(s) if s.isdigit() else 0
        return 0

    def _fetch_video_metadata_pytubefix(self, video_id: str) -> Dict[str, Any]:
        """Fetch basic video metadata via pytubefix.

        This method uses pytubefix's `YouTube` class to load metadata for a single
        video and returns a small dictionary of fields that are useful for
        analytics and enrichment.

        Args:
            video_id (str): YouTube video ID (11-character string).

        Returns:
            Dict[str, Any]: A dictionary with the following keys:
                - "title" (str or None)
                - "duration" (int, seconds; 0 if unavailable)
                - "channelId" (str or None)
                - "author" (str or None)
                - "views" (int or original type if conversion fails)
                - "likes" (int or original type if conversion fails)
                - "description" (str or None)

        Notes:
            Any network or parsing error related to fetching from pytubefix is
            logged as a warning. In that case, the function returns a dict with
            None/0 values for all fields, but the caller continues gracefully.
        """
        url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            yt = YouTube(url)
        except Exception as e:
            # Total failure to construct the YouTube object
            self.logger.warning(f"Failed to fetch pytubefix metadata for {video_id}: {e}")
            return {
                "title": None,
                "duration": 0,
                "channelId": None,
                "author": None,
                "views": None,
                "likes": None,
                "description": None,
            }

        # From here on, we do NOT raise; we only fall back to safe defaults.
        title = getattr(yt, "title", None)
        channel_id = getattr(yt, "channel_id", None)
        author = getattr(yt, "author", None)
        views_raw = getattr(yt, "views", None)
        likes_raw = getattr(yt, "likes", None)
        description = getattr(yt, "description", None)
        length_raw = getattr(yt, "length", None)

        duration = self._safe_int_or_zero(length_raw)
        views = self._normalize_int(views_raw)
        likes = self._normalize_int(likes_raw)

        return {
            "title": title,
            "duration": duration,
            "channelId": channel_id,
            "author": author,
            "views": views,
            "likes": likes,
            "description": description,
        }

    def _ensure_metadata_for_item(self, video_id: str, meta: Dict[str, Any]) -> None:
        """Ensure that a video metadata dictionary is fully populated.

        This method inspects the current metadata for a given video and fills in
        any missing or empty fields using pytubefix.

        Args:
            video_id (str): Video ID corresponding to `meta`.
            meta (Dict[str, Any]): Metadata dictionary for the video. It is
                modified in-place.
        """
        required_fields = ["title", "duration", "channelId", "author", "views", "likes", "description"]

        def _is_missing(val: Any) -> bool:
            if val is None:
                return True
            if isinstance(val, str) and val.strip() == "":
                return True
            return False

        needs_fetch = any(
            (field not in meta or _is_missing(meta.get(field)))
            for field in required_fields
        )
        if not needs_fetch:
            return

        extra = self._fetch_video_metadata_pytubefix(video_id)

        for field in required_fields:
            if field not in meta or _is_missing(meta.get(field)):
                meta[field] = extra.get(field)

    # -------------------------------------------------------------------------
    # ASMR keyword helper
    # -------------------------------------------------------------------------
    def _contains_asmr(self, title: str, description: str = "") -> bool:
        """Check if video content appears to be ASMR based on text fields.

        Args:
            title (str): Video title.
            description (str): Video description. Defaults to an empty string.

        Returns:
            bool: True if `enforce_asmr_keyword` is False, or if either the
            title or description contains the substring "asmr" (case-insensitive).
        """
        if not self.enforce_asmr_keyword:
            return True
        blob = f"{title or ''} {description or ''}".lower()
        return "asmr" in blob

    # -------------------------------------------------------------------------
    # Discovery via YouTube Data API
    # -------------------------------------------------------------------------
    def _discover_with_api(self, seen_ids_set: set, seen_ids_list: List[str]) -> List[Dict[str, Any]]:
        """Discover new ASMR videos using the YouTube Data API.

        Args:
            seen_ids_set (set): A set of already seen video IDs. This set is used
                to avoid processing duplicates. It is modified in-place.
            seen_ids_list (List[str]): A list of seen video IDs preserving insertion
                order. New IDs discovered in this method are appended.

        Returns:
            List[Dict[str, Any]]: A list of newly discovered video metadata dicts.
        """
        if self.youtube is None:
            return []

        new_items: List[Dict[str, Any]] = []
        next_page_token: Optional[str] = None

        for _ in range(self.max_pages):
            request = self.youtube.search().list(  # type: ignore
                q=self.query,
                part="snippet",
                type="video",
                maxResults=self.results_per_page,
                pageToken=next_page_token,
                order="date",
            )
            response = request.execute()

            for item in response.get("items", []):
                video_id = item["id"]["videoId"]
                title = item["snippet"]["title"]

                # Skip if already seen
                if video_id in seen_ids_set:
                    continue

                # Fetch duration details from YouTube Data API
                details = self.youtube.videos().list(  # type: ignore
                    part="contentDetails",
                    id=video_id
                ).execute()

                if not details.get("items"):
                    continue

                iso_duration = details["items"][0]["contentDetails"]["duration"]
                duration_seconds = self._duration_to_seconds(iso_duration)

                # Exclude Shorts
                if self._is_short_video(duration_seconds):
                    continue

                # Enrich with pytubefix metadata
                meta = self._fetch_video_metadata_pytubefix(video_id)
                if not meta.get("duration"):
                    meta["duration"] = duration_seconds

                # ASMR relevance check
                if not self._contains_asmr(title, meta.get("description") or ""):
                    continue

                # Mark as seen and collect
                seen_ids_list.append(video_id)
                seen_ids_set.add(video_id)

                new_items.append({
                    "videoId": video_id,
                    "title": title,
                    "duration": meta.get("duration", duration_seconds),
                    "channelId": meta.get("channelId"),
                    "author": meta.get("author"),
                    "views": meta.get("views"),
                    "likes": meta.get("likes"),
                    "description": meta.get("description"),
                })

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        self.logger.info(f"[API] Found {len(new_items)} new non-short ASMR videos.")
        return new_items

    # -------------------------------------------------------------------------
    # Discovery via pytubefix Search
    # -------------------------------------------------------------------------
    def _discover_with_pytubefix_search(self, seen_ids_set: set, seen_ids_list: List[str]) -> List[Dict[str, Any]]:
        """Discover new ASMR videos using pytubefix contrib Search.

        Args:
            seen_ids_set (set): Set of already seen video IDs. Used to prevent
                duplicates. Modified in-place.
            seen_ids_list (List[str]): Ordered list of seen IDs; new IDs are
                appended to preserve insertion order.

        Returns:
            List[Dict[str, Any]]: List of newly discovered video metadata dicts.
        """
        new_items: List[Dict[str, Any]] = []

        filters = (
            Filter.create()
            .type(Filter.Type.VIDEO)
            .sort_by(Filter.SortBy.RELEVANCE)
        )

        max_results = self.max_pages * self.results_per_page
        count = 0

        self.logger.info("Starting pytubefix Search discovery...")
        s = Search(self.query, filters=filters)

        for v in s.videos:
            if count >= max_results:
                break
            count += 1

            # Extract video ID from search result object or URL
            video_id = getattr(v, "video_id", None) or getattr(v, "videoId", None)
            if not video_id:
                watch_url = getattr(v, "watch_url", "")
                match = re.search(r"v=([0-9A-Za-z_-]{11})", watch_url)
                video_id = match.group(1) if match else None

            if not video_id or video_id in seen_ids_set:
                continue

            title = getattr(v, "title", "") or ""

            # Try to get duration from search object
            try:
                duration_seconds = int(getattr(v, "length", 0))
            except Exception:
                duration_seconds = 0

            # Fetch full metadata via pytubefix
            meta = self._fetch_video_metadata_pytubefix(video_id)

            # If search didn't give a length, use pytubefix metadata duration
            if not duration_seconds and meta.get("duration"):
                duration_seconds = meta["duration"]

            # Exclude Shorts if we have a duration
            if duration_seconds and self._is_short_video(duration_seconds):
                continue

            # ASMR relevance check, prefer title from either search or metadata
            effective_title = title or meta.get("title", "")
            if not self._contains_asmr(effective_title, meta.get("description") or ""):
                continue

            # Mark as seen and collect
            seen_ids_list.append(video_id)
            seen_ids_set.add(video_id)

            new_items.append({
                "videoId": video_id,
                "title": effective_title,
                "duration": duration_seconds,
                "channelId": meta.get("channelId"),
                "author": meta.get("author"),
                "views": meta.get("views"),
                "likes": meta.get("likes"),
                "description": meta.get("description"),
            })

        self.logger.info(f"[pytubefix Search] Found {len(new_items)} new non-short ASMR videos.")
        return new_items

    # -------------------------------------------------------------------------
    # Load existing JSON (new structure only)
    # -------------------------------------------------------------------------
    def _load_existing_by_id(self) -> Dict[str, Dict[str, Any]]:
        """Load existing videos from JSON file into a dict keyed by videoId.

        Assumes JSON is in the unified structure:

            {
                "<videoId>": {
                    "title": "...",
                    "duration": ...,
                    "channelId": "...",
                    "author": "...",
                    "views": ...,
                    "likes": ...,
                    "description": "..."
                },
                ...
            }

        Returns:
            Dict[str, Dict[str, Any]]: Mapping from videoId to its metadata.
        """
        if not os.path.exists(self.json_output):
            return {}

        try:
            with open(self.json_output, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError, TypeError):
            self.logger.warning("Existing JSON could not be parsed; starting fresh.")
            return {}

        if not isinstance(data, dict):
            self.logger.warning("Existing JSON is not a dict; starting fresh.")
            return {}

        existing_by_id: Dict[str, Dict[str, Any]] = {}
        for vid, meta in data.items():
            if not isinstance(meta, dict):
                meta = {}
            existing_by_id[vid] = meta

        return existing_by_id

    # -------------------------------------------------------------------------
    # Main public method
    # -------------------------------------------------------------------------
    def fetch_asmr_videos(self) -> List[str]:
        """Discover and enrich ASMR videos, then persist them to JSON.

        Returns:
            List[str]: A list of newly added video IDs (unique) for this run.
        """
        # 1) Load existing videos and treat them as already known
        existing_by_id = self._load_existing_by_id()
        existing_keys = set(existing_by_id.keys())

        # 2) Load previously seen IDs from file
        seen_ids_list: List[str] = []
        seen_ids_set = set(existing_keys)  # Start with IDs from JSON

        if os.path.exists(self.seen_file):
            with open(self.seen_file, "r", encoding="utf-8") as f:
                for line in f:
                    vid = line.strip()
                    if vid and vid not in seen_ids_set:
                        seen_ids_list.append(vid)
                        seen_ids_set.add(vid)

        # Ensure that all existing JSON IDs appear in the seen IDs list
        for vid in existing_keys:
            if vid not in seen_ids_list:
                seen_ids_list.append(vid)

        all_new_items: List[Dict[str, Any]] = []

        # 3) Discover via YouTube Data API (if enabled)
        if self.youtube is not None:
            api_items = self._discover_with_api(seen_ids_set, seen_ids_list)
            all_new_items.extend(api_items)

        # 4) Discover additional videos via pytubefix Search
        pytube_items = self._discover_with_pytubefix_search(seen_ids_set, seen_ids_list)
        all_new_items.extend(pytube_items)

        self.logger.info(
            f"Total (raw) new non-short ASMR videos discovered this run: {len(all_new_items)}"
        )

        # 5) Merge existing + newly discovered, keyed by videoId
        combined_by_id: Dict[str, Dict[str, Any]] = {}

        # Start with existing metadata
        for vid, meta in existing_by_id.items():
            combined_by_id[vid] = dict(meta) if isinstance(meta, dict) else {}

        # Add or update from new items
        for item in all_new_items:
            vid = item.get("videoId")
            if not vid:
                continue
            meta = combined_by_id.get(vid, {})
            for key, value in item.items():
                if key == "videoId":
                    continue
                meta[key] = value
            combined_by_id[vid] = meta

        # 6) Ensure metadata completeness for all videos
        for vid, meta in combined_by_id.items():
            self._ensure_metadata_for_item(vid, meta)

        # 7) Determine how many video IDs are truly new compared to the original JSON
        final_keys = set(combined_by_id.keys())
        unique_new_ids = final_keys - existing_keys
        num_unique_new = len(unique_new_ids)

        # 8) Save merged results back to JSON in the unified format
        with open(self.json_output, "w", encoding="utf-8") as f:
            json.dump(combined_by_id, f, indent=4, ensure_ascii=False)

        # 9) Update seen IDs file with a stable order
        with open(self.seen_file, "w", encoding="utf-8") as f:
            for vid in seen_ids_list:
                f.write(vid + "\n")

        self.logger.info(
            f"Saved {num_unique_new} unique new entries to '{self.json_output}'. "
            f"Total entries stored: {len(combined_by_id)}"
        )
        self.logger.info(f"Total unique ASMR videos tracked: {len(seen_ids_list)}")

        return list(unique_new_ids)


# -------------------------------------------------------------------------
# Standalone execution entry point
# -------------------------------------------------------------------------
secret = SimpleNamespace(
    API=common.get_secrets("google-api-key")
)

fetcher = ASMRFetcher(
    api_key=secret.API,           # If this is None/empty → only pytubefix Search is used
    query=common.get_configs("query"),
    max_pages=100,
    results_per_page=50,
    seen_file="seen_video_ids.txt",
    json_output="asmr_results.json"
)

if __name__ == "__main__":
    fetcher.fetch_asmr_videos()
