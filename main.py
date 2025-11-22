# -*- coding: utf-8 -*-
"""Keyword-based YouTube video fetcher with metadata enrichment.

This module discovers videos on YouTube for a given query, filters
and enriches them, and persists the results as a JSON file.

It supports two discovery mechanisms:

1. YouTube Data API (if an API key is provided).
2. pytubefix contrib Search (web scraping; no API key required).

Key features:
    * Excludes YouTube Shorts (videos < 60 seconds).
    * Enriches videos with pytubefix metadata:
        - title
        - duration
        - channelId
        - author
        - views
        - likes
        - description
        - uploadDate
        - language
    * Adds channel-level metric (when API is available):
        - channel_average_views (average views per video on that channel)
    * Ensures relevance by requiring the (case-insensitive) query keyword
      to appear in title/description.
    * Deduplicates videos by `videoId`.
    * Persists all video metadata to a JSON file with structure:

        {
            "<videoId>": {
                "title": "...",
                "duration": 1234,
                "channelId": "...",
                "author": "...",
                "views": 12345,
                "likes": 678,
                "description": "...",
                "uploadDate": "2025-01-01T12:34:56Z",
                "language": "en",
                "channel_average_views": 125000.0
            },
            ...
        }

Language detection:
    * Primary source (if using YouTube Data API):
        - snippet.defaultAudioLanguage
        - snippet.defaultLanguage
    * Fallback (for both API + pytubefix-only mode):
        - Automatic detection from title + description using `langdetect`.

Channel average views:
    * Uses YouTube Data API `channels().list(part="statistics")`:
        - statistics.viewCount / statistics.videoCount
    * Caches per-channel results in memory (one API call per channel per run).
    * Stops calling the channels API once quota is exceeded for this run.

Example:
    To run as a script, ensure your configs and secrets are set up,
    then:

        python main.py

Author:
    Shadab Alam <shaadalam.5u@gmail.com>
"""

from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from googleapiclient.discovery import build  # type: ignore
from langdetect import DetectorFactory, detect
from pytubefix import YouTube
from pytubefix.contrib.search import Filter, Search

from logmod import logs
import common
from custom_logger import CustomLogger

# Make langdetect deterministic (otherwise results can vary run-to-run).
DetectorFactory.seed = 0


class ASMRFetcher:
    """Fetch, enrich, and persist query-related videos from YouTube.

    Despite the name, this class can fetch videos for any query, such as
    "ASMR", "Drive", etc. It discovers videos via:

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
          - uploadDate
          - language (auto-detected if not provided by YouTube)
      * Enriches with channel-level metric (when API is available):
          - channel_average_views
      * Ensures relevance by requiring the query keyword (case-insensitive)
        to appear in title/description.
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
                "description": "...",
                "uploadDate": "2025-01-01T12:34:56Z",
                "language": "en",
                "channel_average_views": 125000.0
            },
            ...
        }

    Attributes:
        api_key: YouTube Data API key, or None if not provided.
        query: Search query used for discovery (e.g., "ASMR", "Drive").
        query_keyword: Lowercased version of `query`, used for matching
            in title/description.
        max_pages: Maximum pages to fetch from the YouTube API and as a
            soft cap for pytubefix Search.
        results_per_page: Results per page in the YouTube Data API.
        data_folder: Directory path to hold JSON and seen-IDs file.
        seen_file: Path to the file storing already seen video IDs (one per line).
        json_output: Path to the JSON file storing video metadata.
        youtube: YouTube Data API client or None if no API key is set.
        logger: Custom logger instance for structured logging.
        _channel_stats_cache: In-memory cache for channel statistics lookups.
        _channel_stats_quota_exceeded: Flag set to True once quotaExceeded
            is encountered on channel stats.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        query: str = "ASMR",
        max_pages: int = 100,
        results_per_page: int = 50,
        seen_file: str = "seen_video_ids.txt",
        json_output: str = "asmr_results.json",
    ) -> None:
        """Initialize ASMRFetcher.

        Args:
            api_key: YouTube API key. If None or empty, only pytubefix
                contrib Search is used for discovery.
            query: Search query string used for discovery. This keyword will
                be used for case-insensitive matching in title/description.
            max_pages: Maximum number of pages to fetch in YouTube Data API
                results and also used as a soft cap for pytubefix Search.
            results_per_page: Number of search results per page in the YouTube
                Data API. YouTube usually caps this at 50.
            seen_file: File name (inside the configured data folder) storing
                previously seen video IDs, one per line.
            json_output: File name (inside the configured data folder) storing
                the JSON results.

        Raises:
            KeyError: If `common.get_configs("data")` is missing or misconfigured.
        """
        self.api_key = api_key or None
        self.query = query
        # Use a simple, lowercased keyword for relevance matching.
        self.query_keyword = (self.query or "").strip().lower()

        self.max_pages = max_pages
        self.results_per_page = results_per_page

        # Resolve and ensure data folder exists.
        self.data_folder = common.get_configs("data")
        os.makedirs(self.data_folder, exist_ok=True)

        # Paths for metadata and seen ID tracking, both under data_folder.
        self.seen_file = os.path.join(self.data_folder, seen_file)
        self.json_output = os.path.join(self.data_folder, json_output)

        # Initialize logging.
        logs(show_level=common.get_configs("logger_level"), show_color=True)
        self.logger = CustomLogger(__name__)

        # Initialize YouTube API client only if an API key is given.
        if self.api_key:
            self.youtube = build("youtube", "v3", developerKey=self.api_key)
            self.logger.info("YouTube Data API enabled (API key provided).")
        else:
            self.youtube = None
            self.logger.info("No API key provided. Using only pytubefix Search for discovery.")

        # Simple in-memory cache to avoid repeated channel stats calls.
        # Key: channel_id, Value: channel_average_views (float or None).
        self._channel_stats_cache: Dict[str, Optional[float]] = {}

        # Flag to stop calling channel stats once quota is exceeded in this run.
        self._channel_stats_quota_exceeded: bool = False

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
            iso_duration: ISO 8601 duration string as returned by the YouTube
                Data API in video `contentDetails.duration`.

        Returns:
            Total number of seconds represented by the duration. If the duration
            cannot be parsed, 0 is returned and a warning is logged.
        """
        if iso_duration == "P0D":
            return 0

        # YouTube-specific pattern (PT#H#M#S).
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

        # More generic ISO 8601 pattern if the above fails.
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

        self.logger.warning("Could not parse duration string; using 0 seconds")
        return 0

    def _is_short_video(self, seconds: int) -> bool:
        """Determine whether a video should be considered a YouTube Short.

        Args:
            seconds: Duration of the video in seconds.

        Returns:
            True if `seconds` is strictly less than 60, indicating a short video.
        """
        return seconds < 60

    # -------------------------------------------------------------------------
    # Metadata normalization helpers
    # -------------------------------------------------------------------------
    def _normalize_int(self, val: Any) -> Any:
        """Normalize numeric fields (views/likes) to integers when possible.

        Args:
            val: Value to normalize, possibly int, float, or string.

        Returns:
            Integer if the value can be safely converted; otherwise the original
            value (unchanged).
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

        Intended for duration-like fields where failure should not raise.

        Args:
            val: Value that may represent an integer.

        Returns:
            Parsed integer value, or 0 if parsing is not possible.
        """
        if isinstance(val, int):
            return val
        if isinstance(val, float):
            return int(val)
        if isinstance(val, str):
            s = val.replace(",", "").strip()
            return int(s) if s.isdigit() else 0
        return 0

    def _normalize_upload_date(self, val: Any) -> Optional[str]:
        """Normalize upload date to a string, if possible.

        This method accepts common date-like values (e.g. datetime.date,
        datetime.datetime, string) and returns an ISO-formatted string when
        possible.

        Args:
            val: Raw upload date value.

        Returns:
            Normalized upload date string or None if not available.
        """
        if val is None:
            return None

        try:
            import datetime

            if isinstance(val, (datetime.date, datetime.datetime)):
                return val.isoformat()
        except Exception:
            # Fall back to generic conversion.
            pass

        try:
            return str(val)
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Language detection helper
    # -------------------------------------------------------------------------
    def _detect_language(self, text: str) -> Optional[str]:
        """Detect the language of the given text.

        Uses the `langdetect` library to infer the language code from free-form
        text (e.g. title + description).

        Args:
            text: Input text used for language detection.

        Returns:
            BCP-47-like language code (e.g. "en", "fr", "de") if detection
            succeeds; otherwise None.
        """
        text = (text or "").strip()
        if not text:
            return None
        try:
            return detect(text)
        except Exception:
            # Detection can fail on very short or noisy text.
            return None

    # -------------------------------------------------------------------------
    # Channel statistics helper
    # -------------------------------------------------------------------------
    def _fetch_channel_average_views(self, channel_id: str) -> Optional[float]:
        """Fetch average views per video for a given channel using the YouTube Data API.

        This method queries the channel statistics and computes:

            channel_average_views = viewCount / videoCount

        Results are cached in-memory per channel ID to avoid repeated API calls
        during a single run.

        Args:
            channel_id: The YouTube channel ID.

        Returns:
            Average views per video as a float, or None if unavailable or
            if the YouTube API client is not configured or quota is exceeded.
        """
        if not channel_id:
            return None

        # If we've already decided not to call the API anymore this run, bail out.
        if self._channel_stats_quota_exceeded:
            return None

        # Return from cache if present (even if cached None).
        if channel_id in self._channel_stats_cache:
            return self._channel_stats_cache[channel_id]

        if self.youtube is None:
            # Cannot compute without YouTube Data API.
            self._channel_stats_cache[channel_id] = None
            return None

        try:
            response = self.youtube.channels().list(  # type: ignore[call-arg]
                part="statistics",
                id=channel_id,
            ).execute()

            items = response.get("items", [])
            if not items:
                self._channel_stats_cache[channel_id] = None
                return None

            stats = items[0].get("statistics", {})
            total_views = int(stats.get("viewCount", 0))
            total_videos = int(stats.get("videoCount", 0))

            if total_videos <= 0:
                avg = None
            else:
                avg = total_views / float(total_videos)

            # Cache result (including None) so we do not refetch this run.
            self._channel_stats_cache[channel_id] = avg
            return avg

        except Exception as exc:  # noqa: BLE001
            # Do NOT put exc string into the format message (it contains braces).
            self.logger.warning(
                "Failed to fetch channel statistics; channel_average_views will be None for this channel"
            )

            # If this looks like a quotaExceeded error, remember it so we
            # do not hammer the API with more failing requests this run.
            if "quotaExceeded" in str(exc):
                self._channel_stats_quota_exceeded = True
                self.logger.warning(
                    "YouTube channel statistics quota exceeded; "
                    "channel_average_views will be skipped for the rest of this run"
                )

            self._channel_stats_cache[channel_id] = None
            return None

    # -------------------------------------------------------------------------
    # pytubefix metadata helpers
    # -------------------------------------------------------------------------
    def _fetch_video_metadata_pytubefix(self, video_id: str) -> Dict[str, Any]:
        """Fetch basic video metadata via pytubefix.

        This method uses pytubefix's `YouTube` class to load metadata for a single
        video and returns a dictionary of fields that are useful for analytics
        and enrichment.

        Args:
            video_id: YouTube video ID (11-character string).

        Returns:
            A dictionary with the following keys:
                - "title" (str or None)
                - "duration" (int, seconds; 0 if unavailable)
                - "channelId" (str or None)
                - "author" (str or None)
                - "views" (int or original type if conversion fails)
                - "likes" (int or original type if conversion fails)
                - "description" (str or None)
                - "uploadDate" (str or None)
                - "language" (str or None, auto-detected from text)
                - "channel_average_views" (None here; filled later if API is available)

            In case of a network or parsing error, the function logs a warning
            and returns a dict with default values.
        """
        url = "https://www.youtube.com/watch?v=" + video_id

        try:
            yt = YouTube(url)
        except Exception:  # noqa: BLE001
            # Total failure to construct the YouTube object.
            self.logger.warning(
                "Failed to fetch pytubefix metadata; using default values for this video"
            )
            return {
                "title": None,
                "duration": 0,
                "channelId": None,
                "author": None,
                "views": None,
                "likes": None,
                "description": None,
                "uploadDate": None,
                "language": None,
                "channel_average_views": None,
            }

        # From here on, we do NOT raise; we only fall back to safe defaults.
        title = getattr(yt, "title", None)
        channel_id = getattr(yt, "channel_id", None)
        author = getattr(yt, "author", None)
        views_raw = getattr(yt, "views", None)
        likes_raw = getattr(yt, "likes", None)
        description = getattr(yt, "description", None)
        length_raw = getattr(yt, "length", None)
        publish_raw = getattr(yt, "publish_date", None)

        duration = self._safe_int_or_zero(length_raw)
        views = self._normalize_int(views_raw)
        likes = self._normalize_int(likes_raw)
        upload_date = self._normalize_upload_date(publish_raw)

        # Try to detect language from title + description.
        combined_text = (title or "") + " " + (description or "")
        language = self._detect_language(combined_text)

        # We do NOT compute channel_average_views here; that is done via API later.
        return {
            "title": title,
            "duration": duration,
            "channelId": channel_id,
            "author": author,
            "views": views,
            "likes": likes,
            "description": description,
            "uploadDate": upload_date,
            "language": language,
            "channel_average_views": None,
        }

    def _ensure_metadata_for_item(self, video_id: str, meta: Dict[str, Any]) -> None:
        """Ensure that a video metadata dictionary is fully populated.

        This method inspects the current metadata for a given video and fills in
        any missing or empty fields using pytubefix. Language is handled with
        a separate lightweight detection step. Channel average views are
        computed via the YouTube Data API if possible.

        Args:
            video_id: Video ID corresponding to `meta`.
            meta: Metadata dictionary for the video. It is modified in-place.
        """
        required_fields = [
            "title",
            "duration",
            "channelId",
            "author",
            "views",
            "likes",
            "description",
            "uploadDate",
        ]

        def _is_missing(val: Any) -> bool:
            if val is None:
                return True
            if isinstance(val, str) and val.strip() == "":
                return True
            return False

        needs_fetch = any(
            (field not in meta) or _is_missing(meta.get(field))
            for field in required_fields
        )

        if needs_fetch:
            # Re-fetch from pytubefix if any of the core fields are missing.
            extra = self._fetch_video_metadata_pytubefix(video_id)
            for field in required_fields:
                if field not in meta or _is_missing(meta.get(field)):
                    meta[field] = extra.get(field)

            # Only overwrite language with pytubefix-derived language if we do not have one yet.
            if not meta.get("language"):
                meta["language"] = extra.get("language")

        # Language fallback: detect from title + description if still missing.
        if not meta.get("language"):
            combined_text = (meta.get("title", "") or "") + " " + (meta.get("description", "") or "")
            meta["language"] = self._detect_language(combined_text)

        # Channel average views: compute via YouTube Data API if missing and channelId is known.
        # We only do this when the field is None or absent, so existing values are never overwritten.
        if ("channel_average_views" not in meta) or (meta.get("channel_average_views") is None):
            channel_id = meta.get("channelId")
            if channel_id:
                meta["channel_average_views"] = self._fetch_channel_average_views(channel_id)

    # -------------------------------------------------------------------------
    # Keyword relevance helper
    # -------------------------------------------------------------------------
    def _contains_query_keyword(self, title: str, description: str = "") -> bool:
        """Check if video content appears relevant based on the query keyword.

        The check is case-insensitive and uses a simple substring search
        over the concatenation of title and description.

        Example:
            If query="ASMR", this matches "asmr", "ASMR", "Asmr", etc.
            If query="Drive", this matches "drive", "Drive", "DRIVE", etc.

        Args:
            title: Video title.
            description: Video description.

        Returns:
            True if the lowercased query keyword is found in either the title
            or description; False otherwise. If `query_keyword` is empty,
            this returns True (no filtering).
        """
        if not self.query_keyword:
            # If no keyword is set, skip filtering.
            return True

        blob = (title or "") + " " + (description or "")
        blob = blob.lower()
        return self.query_keyword in blob

    # -------------------------------------------------------------------------
    # Discovery via YouTube Data API
    # -------------------------------------------------------------------------
    def _discover_with_api(
        self,
        seen_ids_set: set[str],
        seen_ids_list: List[str],
    ) -> List[Dict[str, Any]]:
        """Discover new relevant videos using the YouTube Data API.

        Args:
            seen_ids_set: A set of already seen video IDs. This set is used to
                avoid processing duplicates and is modified in-place.
            seen_ids_list: A list of seen video IDs preserving insertion order.
                New IDs discovered in this method are appended.

        Returns:
            A list of newly discovered video metadata dictionaries. If the
            quota is exceeded or the API call fails, an empty list is returned
            and the method logs a warning, but does not raise.
        """
        if self.youtube is None:
            return []

        new_items: List[Dict[str, Any]] = []
        next_page_token: Optional[str] = None

        for _ in range(self.max_pages):
            try:
                request = self.youtube.search().list(  # type: ignore[call-arg]
                    q=self.query,
                    part="snippet",
                    type="video",
                    maxResults=self.results_per_page,
                    pageToken=next_page_token,
                    order="date",
                )
                response = request.execute()
            except Exception as exc:  # noqa: BLE001
                # We do not include exc text as it contains braces.
                self.logger.warning(
                    "YouTube Data API search failed; falling back to pytubefix only for this run"
                )
                if "quotaExceeded" in str(exc):
                    self.logger.warning(
                        "YouTube search quota exceeded; skipping further API search calls this run"
                    )
                return []

            for item in response.get("items", []):
                video_id = item["id"]["videoId"]
                snippet = item["snippet"]
                title = snippet["title"]
                upload_date_api = snippet.get("publishedAt")

                # YouTube snippet language fields.
                default_lang = snippet.get("defaultAudioLanguage") or snippet.get(
                    "defaultLanguage"
                )

                # Skip if already seen.
                if video_id in seen_ids_set:
                    continue

                # Fetch duration details from YouTube Data API.
                try:
                    details = self.youtube.videos().list(  # type: ignore[call-arg]
                        part="contentDetails",
                        id=video_id,
                    ).execute()
                except Exception:
                    # If duration fetch fails, skip this video.
                    continue

                if not details.get("items"):
                    continue

                content_details = details["items"][0].get("contentDetails", {})
                iso_duration = content_details.get("duration", "P0D")
                duration_seconds = self._duration_to_seconds(iso_duration)

                # Exclude Shorts.
                if self._is_short_video(duration_seconds):
                    continue

                # Enrich with pytubefix metadata.
                meta = self._fetch_video_metadata_pytubefix(video_id)
                if not meta.get("duration"):
                    meta["duration"] = duration_seconds

                # Relevance check, using API title and pytubefix description.
                if not self._contains_query_keyword(
                    title, meta.get("description") or ""
                ):
                    continue

                # Prefer API upload date, fall back to pytubefix upload date.
                upload_date_meta = meta.get("uploadDate")
                final_upload_date = upload_date_api or upload_date_meta

                # Prefer API language field, fall back to pytubefix / detected language.
                meta_language = meta.get("language")
                final_language = default_lang or meta_language

                # Channel average views via stats endpoint (safe, cached).
                channel_id = meta.get("channelId")
                if channel_id:
                    channel_avg_views = self._fetch_channel_average_views(channel_id)
                else:
                    channel_avg_views = None

                # Mark as seen and collect.
                seen_ids_list.append(video_id)
                seen_ids_set.add(video_id)

                new_items.append(
                    {
                        "videoId": video_id,
                        "title": title,
                        "duration": meta.get("duration", duration_seconds),
                        "channelId": channel_id,
                        "author": meta.get("author"),
                        "views": meta.get("views"),
                        "likes": meta.get("likes"),
                        "description": meta.get("description"),
                        "uploadDate": final_upload_date,
                        "language": final_language,
                        "channel_average_views": channel_avg_views,
                    }
                )

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        self.logger.info(
            "[API] Found {} new non-short relevant videos.".format(len(new_items))
        )
        return new_items

    # -------------------------------------------------------------------------
    # Discovery via pytubefix Search
    # -------------------------------------------------------------------------
    def _discover_with_pytubefix_search(
        self,
        seen_ids_set: set[str],
        seen_ids_list: List[str],
    ) -> List[Dict[str, Any]]:
        """Discover new relevant videos using pytubefix contrib Search.

        Args:
            seen_ids_set: Set of already seen video IDs. Used to prevent
                duplicates. Modified in-place.
            seen_ids_list: Ordered list of seen IDs; new IDs are appended to
                preserve insertion order.

        Returns:
            List of newly discovered video metadata dictionaries.
        """
        new_items: List[Dict[str, Any]] = []

        # Restrict results to videos and sort by relevance.
        filters = (
            Filter.create()
            .type(Filter.Type.VIDEO)
            .sort_by(Filter.SortBy.RELEVANCE)
        )

        max_results = self.max_pages * self.results_per_page
        count = 0

        self.logger.info("Starting pytubefix Search discovery...")
        search_obj = Search(self.query, filters=filters)

        for v in search_obj.videos:
            if count >= max_results:
                break
            count += 1

            # Extract video ID from search result object or URL.
            video_id = getattr(v, "video_id", None) or getattr(v, "videoId", None)
            if not video_id:
                watch_url = getattr(v, "watch_url", "") or ""
                match = re.search(r"v=([0-9A-Za-z_-]{11})", watch_url)
                video_id = match.group(1) if match else None

            if not video_id or video_id in seen_ids_set:
                continue

            title = getattr(v, "title", "") or ""

            # Try to get duration from search object.
            try:
                duration_seconds = int(getattr(v, "length", 0))
            except Exception:  # noqa: BLE001
                duration_seconds = 0

            # Fetch full metadata via pytubefix to enrich.
            meta = self._fetch_video_metadata_pytubefix(video_id)

            # If search didn't give a length, use pytubefix metadata duration.
            if not duration_seconds and meta.get("duration"):
                duration_seconds = meta["duration"]

            # Exclude Shorts if we have a duration.
            if duration_seconds and self._is_short_video(duration_seconds):
                continue

            # Relevance check; prefer title from either search or metadata.
            effective_title = title or meta.get("title", "") or ""
            if not self._contains_query_keyword(
                effective_title, meta.get("description") or ""
            ):
                continue

            # Channel average views via API, if possible (cached).
            channel_id = meta.get("channelId")
            if channel_id:
                channel_avg_views = self._fetch_channel_average_views(channel_id)
            else:
                channel_avg_views = None

            # Mark as seen and collect.
            seen_ids_list.append(video_id)
            seen_ids_set.add(video_id)

            new_items.append(
                {
                    "videoId": video_id,
                    "title": effective_title,
                    "duration": duration_seconds,
                    "channelId": channel_id,
                    "author": meta.get("author"),
                    "views": meta.get("views"),
                    "likes": meta.get("likes"),
                    "description": meta.get("description"),
                    "uploadDate": meta.get("uploadDate"),
                    "language": meta.get("language"),
                    "channel_average_views": channel_avg_views,
                }
            )

        self.logger.info(
            "[pytubefix Search] Found {} new non-short relevant videos.".format(
                len(new_items)
            )
        )
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
                    "description": "...",
                    "uploadDate": "...",
                    "language": "...",
                    "channel_average_views": ...
                },
                ...
            }

        Returns:
            Mapping from videoId to its metadata. If the JSON file does not
            exist or cannot be parsed, an empty dict is returned.
        """
        if not os.path.exists(self.json_output):
            return {}

        try:
            with open(self.json_output, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError, TypeError):
            self.logger.warning("Existing JSON could not be parsed; starting fresh")
            return {}

        if not isinstance(data, dict):
            self.logger.warning("Existing JSON is not a dict; starting fresh")
            return {}

        existing_by_id: Dict[str, Dict[str, Any]] = {}
        for vid, meta in data.items():
            if not isinstance(meta, dict):
                meta = {}
            existing_by_id[str(vid)] = meta

        return existing_by_id

    # -------------------------------------------------------------------------
    # Main public method
    # -------------------------------------------------------------------------
    def fetch_asmr_videos(self) -> List[str]:
        """Discover and enrich videos, then persist them to JSON.

        The method name is historical and kept for backward compatibility; it
        works for any query (not just ASMR).

        This method orchestrates:
            1. Loading existing video metadata and seen IDs.
            2. Discovering new videos via the YouTube Data API (if enabled).
            3. Discovering additional videos via pytubefix Search.
            4. Ensuring metadata completeness **only for newly added videos**.
            5. Writing a unified JSON file keyed by videoId.
            6. Updating the `seen_video_ids.txt` file.

        Returns:
            A list of newly added unique video IDs for this run.
        """
        # 1) Load existing videos and treat them as already known.
        existing_by_id = self._load_existing_by_id()
        existing_keys = set(existing_by_id.keys())

        # 2) Load previously seen IDs from file (if present).
        seen_ids_list: List[str] = []
        seen_ids_set: set[str] = set(existing_keys)  # Start with IDs from JSON.

        if os.path.exists(self.seen_file):
            with open(self.seen_file, "r", encoding="utf-8") as f:
                for line in f:
                    vid = line.strip()
                    if vid and vid not in seen_ids_set:
                        seen_ids_list.append(vid)
                        seen_ids_set.add(vid)

        # Ensure that all existing JSON IDs appear in the seen IDs list.
        for vid in existing_keys:
            if vid not in seen_ids_list:
                seen_ids_list.append(vid)

        all_new_items: List[Dict[str, Any]] = []

        # 3) Discover via YouTube Data API (if enabled).
        if self.youtube is not None:
            api_items = self._discover_with_api(seen_ids_set, seen_ids_list)
            all_new_items.extend(api_items)

        # 4) Discover additional videos via pytubefix Search.
        pytube_items = self._discover_with_pytubefix_search(seen_ids_set, seen_ids_list)
        all_new_items.extend(pytube_items)

        self.logger.info(
            "Total new non-short relevant videos discovered this run: {}".format(
                len(all_new_items)
            )
        )

        # 5) Merge existing + newly discovered, keyed by videoId.
        combined_by_id: Dict[str, Dict[str, Any]] = {}

        # Start with existing metadata.
        for vid, meta in existing_by_id.items():
            combined_by_id[vid] = dict(meta) if isinstance(meta, dict) else {}

        # Add or update from new items.
        for item in all_new_items:
            vid = item.get("videoId")
            if not vid:
                continue
            meta = combined_by_id.get(vid, {})
            # Merge fields from the newly discovered item.
            for key, value in item.items():
                if key == "videoId":
                    continue
                meta[key] = value
            combined_by_id[vid] = meta

        # 6) Ensure metadata completeness **only for newly added videos**.
        final_keys = set(combined_by_id.keys())
        unique_new_ids = final_keys - existing_keys
        self.logger.info(
            "Ensuring metadata for {} newly added videos".format(len(unique_new_ids))
        )

        for vid in unique_new_ids:
            self._ensure_metadata_for_item(vid, combined_by_id[vid])

        # 7) Save merged results back to JSON in the unified format.
        with open(self.json_output, "w", encoding="utf-8") as f:
            json.dump(combined_by_id, f, indent=4, ensure_ascii=False)

        # 8) Update seen IDs file with a stable order.
        with open(self.seen_file, "w", encoding="utf-8") as f:
            for vid in seen_ids_list:
                f.write(vid + "\n")

        self.logger.info(
            "Saved {} unique new entries to '{}'. Total entries stored: {}".format(
                len(unique_new_ids), self.json_output, len(combined_by_id)
            )
        )
        self.logger.info(
            "Total unique videos tracked (seen list size): {}".format(len(seen_ids_list))
        )

        return list(unique_new_ids)


# -------------------------------------------------------------------------
# Standalone execution entry point
# -------------------------------------------------------------------------
secret = SimpleNamespace(
    API=common.get_secrets("google-api-key"),
)

fetcher = ASMRFetcher(
    api_key=secret.API,  # If this is None/empty → only pytubefix Search is used.
    query=common.get_configs("query"),
    max_pages=100,
    results_per_page=50,
    seen_file="seen_video_ids.txt",
    json_output="asmr_results.json",
)

if __name__ == "__main__":
    fetcher.fetch_asmr_videos()
