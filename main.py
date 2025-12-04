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
      to appear in the **title**.
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

Date filtering:
    * Supports both publishedBefore and publishedAfter:
        - If both are None → no date filter.
        - If only publishedBefore → keep videos strictly before that time.
        - If only publishedAfter → keep videos strictly after that time.
        - If both → keep videos strictly between them.
    * Values may be:
        - None, empty, "none" → treated as no filter.
        - "YYYY-MM-DD" → converted to "YYYY-MM-DDT00:00:00Z".
        - Any other string is assumed to be RFC3339 and passed through.

Date windowing:
    * Optional config `date_window_months` (integer, in months).
    * Uses `date_before` (or legacy `date`) as global **start**.
    * Uses `date_after` as global **end**.
    * Splits [date_before, date_after] into consecutive month windows:
        - Run 1:  start = date_before,   end = start + window
        - Run 2:  start = previous end,  end = start + window
        - ...
        - Last run is clamped so end <= date_after.
    * Once the start reaches or passes `date_after`, no further
      window is produced and **no date filter is passed** (so we do
      not “pass this date” again).

Example:
    To run as a script, ensure your configs and secrets are set up,
    then:

        python main.py

Author:
    Shadab Alam <md_shadab_alam@outlook.com>
"""

from __future__ import annotations

import json
import os
import re
import random
import datetime
import calendar
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from googleapiclient.discovery import build  # type: ignore
from langdetect import DetectorFactory, detect
from pytubefix import YouTube, exceptions as pytube_exceptions
from pytubefix.contrib.search import Filter, Search

from logmod import logs
import common
from custom_logger import CustomLogger

# Make langdetect deterministic (otherwise results can vary run-to-run).
DetectorFactory.seed = 0


class ASMRFetcher:
    """Fetch, enrich, and persist query-related videos from YouTube."""

    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        query: str = "ASMR",
        max_pages: int = 100,
        results_per_page: int = 50,
        seen_file: str = "seen_video_ids.txt",
        json_output: str = "asmr_results.json",
        published_before: Optional[str] = None,
        published_after: Optional[str] = None,
    ) -> None:
        """Initialize ASMRFetcher."""
        # Core configs
        self.query = query
        self.query_keyword = (self.query or "").strip().lower()
        self.max_pages = max_pages
        self.results_per_page = results_per_page

        # Paths
        self.data_folder = common.get_configs("data")
        os.makedirs(self.data_folder, exist_ok=True)
        self.seen_file = os.path.join(self.data_folder, seen_file)
        self.json_output = os.path.join(self.data_folder, json_output)

        # Logging
        logs(show_level=common.get_configs("logger_level"), show_color=True)
        self.logger = CustomLogger(__name__)

        # Date filters
        self.published_before = self._normalize_published_bound(published_before)
        self.published_after = self._normalize_published_bound(published_after)

        # Multiple API keys support
        raw_keys = api_keys or []
        self.api_keys: List[str] = [k for k in raw_keys if k]
        self._current_key_index: int = 0
        self.youtube = None

        # Caches / flags
        self._channel_stats_cache: Dict[str, Optional[float]] = {}
        self._channel_stats_quota_exceeded: bool = False
        self._pytube_disabled: bool = False

        # Initialize YouTube client (if any API key exists)
        if self.api_keys:
            self._init_youtube_for_current_key()
        else:
            self.logger.info(
                "No YouTube Data API key provided. Using only pytubefix Search for discovery."
            )

    # -------------------------------------------------------------------------
    # API key handling
    # -------------------------------------------------------------------------
    def _init_youtube_for_current_key(self) -> None:
        """Initialize YouTube client for the current API key index."""
        if not self.api_keys:
            self.youtube = None
            return

        key = self.api_keys[self._current_key_index]
        try:
            self.youtube = build(
                "youtube",
                "v3",
                developerKey=key,
                cache_discovery=False,
            )
            total_keys = len(self.api_keys)
            current = self._current_key_index + 1
            self.logger.info(
                f"YouTube Data API enabled with {total_keys} API key(s). Using key {current}/{total_keys}."
            )
            if self.published_before or self.published_after:
                self.logger.info(
                    f"Using date filter: published_after={self.published_after}, "
                    f"published_before={self.published_before}"
                )
            else:
                self.logger.info("No date filters set; fetching normally (all dates).")
        except Exception:
            self.youtube = None
            idx = self._current_key_index + 1
            self.logger.warning(
                f"Failed to initialize YouTube Data API client for key index {idx}."
            )
            # Try to switch to the next key immediately.
            self._switch_to_next_api_key()

    def _switch_to_next_api_key(self) -> bool:
        """
        Rotate to the next API key.

        Returns True if a new key was successfully initialized,
        False if there are no more keys or initialization fails.
        """
        if not self.api_keys:
            self.youtube = None
            return False

        self._current_key_index += 1
        if self._current_key_index >= len(self.api_keys):
            self.youtube = None
            self.logger.warning(
                "All YouTube Data API keys have been exhausted or failed. "
                "Continuing without Data API for the rest of this run."
            )
            return False

        key = self.api_keys[self._current_key_index]
        try:
            self.youtube = build(
                "youtube",
                "v3",
                developerKey=key,
                cache_discovery=False,
            )
            self._channel_stats_quota_exceeded = False
            total_keys = len(self.api_keys)
            current = self._current_key_index + 1
            self.logger.info(
                f"Switched to YouTube Data API key {current}/{total_keys}."
            )
            if self.published_before or self.published_after:
                self.logger.info(
                    f"Using date filter: published_after={self.published_after}, "
                    f"published_before={self.published_before}"
                )
            return True
        except Exception:
            idx = self._current_key_index + 1
            self.logger.warning(
                f"Failed to initialize YouTube Data API client for key index {idx}; trying next key."
            )
            # Try the next key recursively until we run out.
            return self._switch_to_next_api_key()

    # -------------------------------------------------------------------------
    # Small helpers
    # -------------------------------------------------------------------------
    def _empty_video_metadata(self) -> Dict[str, Any]:
        """Return an empty/default metadata dict for a video."""
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

    def _normalize_published_bound(self, value: Optional[str]) -> Optional[str]:
        """
        Normalize a user-provided date bound (publishedBefore/After) to RFC3339.

        Rules:
            - None, empty string, or 'none' (case-insensitive) -> None (no filter)
            - 'YYYY-MM-DD' -> 'YYYY-MM-DDT00:00:00Z'
            - Anything else is passed through unchanged (assumed valid RFC3339)
        """
        if not value:
            return None

        v = value.strip()
        if not v or v.lower() == "none":
            return None

        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            return f"{v}T00:00:00Z"

        return v

    def _parse_iso_like_datetime(self, value: str):
        """
        Parse an ISO-ish datetime string into a datetime object.
        Accepts 'YYYY-MM-DD', 'YYYY-MM-DDTHH:MM:SS', '...Z', etc.
        """
        if not value:
            return None

        try:
            s = str(value).strip()
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            return datetime.datetime.fromisoformat(s)
        except Exception:
            return None

    def _passes_date_filter(self, upload_date: Optional[str]) -> bool:
        """
        Return True if the video with the given upload_date should be kept
        under the current published_before / published_after filters.
        """
        if self.published_before is None and self.published_after is None:
            return True

        if not upload_date:
            return True

        pb_dt = self._parse_iso_like_datetime(self.published_before) if self.published_before else None
        pa_dt = self._parse_iso_like_datetime(self.published_after) if self.published_after else None
        up_dt = self._parse_iso_like_datetime(upload_date)

        if up_dt is None:
            return True

        if up_dt.tzinfo is not None:
            up_dt = up_dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        if pb_dt is not None and pb_dt.tzinfo is not None:
            pb_dt = pb_dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        if pa_dt is not None and pa_dt.tzinfo is not None:
            pa_dt = pa_dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)

        if pb_dt is None and pa_dt is not None:
            return up_dt > pa_dt
        if pa_dt is None and pb_dt is not None:
            return up_dt < pb_dt
        if pa_dt is not None and pb_dt is not None:
            return pa_dt < up_dt < pb_dt

        return True

    # -------------------------------------------------------------------------
    # Duration helpers
    # -------------------------------------------------------------------------
    def _duration_to_seconds(self, iso_duration: str) -> int:
        """Convert an ISO 8601 YouTube duration string to total seconds."""
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

        self.logger.warning("Could not parse duration string; using 0 seconds")
        return 0

    def _is_short_video(self, seconds: int) -> bool:
        """Determine whether a video should be considered a YouTube Short."""
        return seconds < 60

    # -------------------------------------------------------------------------
    # Metadata normalization helpers
    # -------------------------------------------------------------------------
    def _normalize_int(self, val: Any) -> Any:
        """Normalize numeric fields (views/likes) to integers when possible."""
        if isinstance(val, int):
            return val
        if isinstance(val, float):
            return int(val)
        if isinstance(val, str):
            s = val.replace(",", "").strip()
            return int(s) if s.isdigit() else val
        return val

    def _safe_int_or_zero(self, val: Any) -> int:
        """Convert a value to int if reasonable, otherwise return 0."""
        if isinstance(val, int):
            return val
        if isinstance(val, float):
            return int(val)
        if isinstance(val, str):
            s = val.replace(",", "").strip()
            return int(s) if s.isdigit() else 0
        return 0

    def _normalize_upload_date(self, val: Any) -> Optional[str]:
        """Normalize upload date to a string, if possible."""
        if val is None:
            return None

        try:
            if isinstance(val, (datetime.date, datetime.datetime)):
                return val.isoformat()
        except Exception:
            pass

        try:
            return str(val)
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Language detection helper
    # -------------------------------------------------------------------------
    def _detect_language(self, text: str) -> Optional[str]:
        """Detect the language of the given text using langdetect."""
        text = (text or "").strip()
        if not text:
            return None
        try:
            return detect(text)
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Channel statistics helper
    # -------------------------------------------------------------------------
    def _fetch_channel_average_views(self, channel_id: str) -> Optional[float]:
        """Fetch average views per video for a given channel using the YouTube Data API."""
        if not channel_id:
            return None

        if self._channel_stats_quota_exceeded:
            return None

        if channel_id in self._channel_stats_cache:
            return self._channel_stats_cache[channel_id]

        if self.youtube is None:
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

            self._channel_stats_cache[channel_id] = avg
            return avg

        except Exception as exc:  # noqa: BLE001
            if "quotaExceeded" in str(exc):
                self.logger.warning(
                    "YouTube channel statistics quota exceeded for current key."
                )
                # Try next key if available
                if self._switch_to_next_api_key():
                    return self._fetch_channel_average_views(channel_id)
                else:
                    self._channel_stats_quota_exceeded = True
                    self._channel_stats_cache[channel_id] = None
                    return None

            self.logger.warning(
                "Failed to fetch channel statistics; channel_average_views will be None for this channel"
            )
            self._channel_stats_cache[channel_id] = None
            return None

    # -------------------------------------------------------------------------
    # pytubefix metadata helpers
    # -------------------------------------------------------------------------
    def _fetch_video_metadata_pytubefix(self, video_id: str) -> Dict[str, Any]:
        """Fetch basic video metadata via pytubefix (defensive, BotDetection-aware)."""
        if self._pytube_disabled:
            return self._empty_video_metadata()

        url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            yt = YouTube(url, "WEB")
        except pytube_exceptions.BotDetection:
            self.logger.warning(
                "pytubefix BotDetection when constructing YouTube object; "
                "disabling pytubefix for the rest of this run"
            )
            self._pytube_disabled = True
            return self._empty_video_metadata()
        except Exception:
            self.logger.warning(
                "Failed to construct pytubefix YouTube object; using default metadata for this video"
            )
            return self._empty_video_metadata()

        try:
            title = getattr(yt, "title", None)
            channel_id = getattr(yt, "channel_id", None)
            author = getattr(yt, "author", None)
            views_raw = getattr(yt, "views", None)
            likes_raw = getattr(yt, "likes", None)
            description = getattr(yt, "description", None)

            try:
                length_raw = getattr(yt, "length", None)
            except Exception:
                length_raw = None

            publish_raw = getattr(yt, "publish_date", None)

            duration = self._safe_int_or_zero(length_raw)
            views = self._normalize_int(views_raw)
            likes = self._normalize_int(likes_raw)
            upload_date = self._normalize_upload_date(publish_raw)

            combined_text = (title or "") + " " + (description or "")
            language = self._detect_language(combined_text)

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

        except pytube_exceptions.BotDetection:
            self.logger.warning(
                "pytubefix BotDetection when extracting metadata; "
                "disabling pytubefix for the rest of this run"
            )
            self._pytube_disabled = True
            return self._empty_video_metadata()
        except Exception:
            self.logger.warning(
                "Failed to extract pytubefix metadata; using default metadata for this video"
            )
            return self._empty_video_metadata()

    def _ensure_metadata_for_item(self, video_id: str, meta: Dict[str, Any]) -> None:
        """Ensure that a video metadata dictionary is fully populated."""
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

        if needs_fetch and not self._pytube_disabled:
            extra = self._fetch_video_metadata_pytubefix(video_id)
            for field in required_fields:
                if field not in meta or _is_missing(meta.get(field)):
                    meta[field] = extra.get(field)

            if not meta.get("language"):
                meta["language"] = extra.get("language")

        if not meta.get("language"):
            combined_text = (meta.get("title", "") or "") + " " + (meta.get("description", "") or "")
            meta["language"] = self._detect_language(combined_text)

        if ("channel_average_views" not in meta) or (meta.get("channel_average_views") is None):
            channel_id = meta.get("channelId")
            if channel_id:
                meta["channel_average_views"] = self._fetch_channel_average_views(channel_id)

    # -------------------------------------------------------------------------
    # Keyword relevance helper
    # -------------------------------------------------------------------------
    def _contains_query_keyword(self, title: str, description: str = "") -> bool:
        """Check if video content appears relevant based on the query keyword.

        NOTE: Only the title is checked. Description is ignored on purpose.
        """
        if not self.query_keyword:
            return True
        return self.query_keyword in (title or "").lower()

    # -------------------------------------------------------------------------
    # Discovery via YouTube Data API (with multi-key rotation)
    # -------------------------------------------------------------------------
    def _discover_with_api(self, seen_ids_set: set[str], seen_ids_list: List[str]) -> List[Dict[str, Any]]:
        """Discover new relevant videos using the YouTube Data API."""
        if self.youtube is None:
            return []

        new_items: List[Dict[str, Any]] = []
        next_page_token: Optional[str] = None

        for _ in range(self.max_pages):
            if self.youtube is None:
                break

            try:
                search_params: Dict[str, Any] = {
                    "q": self.query,
                    "part": "snippet",
                    "type": "video",
                    "maxResults": self.results_per_page,
                }

                if next_page_token:
                    search_params["pageToken"] = next_page_token

                if self.published_before is not None:
                    search_params["publishedBefore"] = self.published_before
                if self.published_after is not None:
                    search_params["publishedAfter"] = self.published_after

                request = self.youtube.search().list(  # type: ignore[call-arg]
                    **search_params
                )
                response = request.execute()
            except Exception as exc:  # noqa: BLE001
                if "quotaExceeded" in str(exc):
                    self.logger.warning(
                        "YouTube search quota exceeded for current key; attempting to switch API key."
                    )
                    if self._switch_to_next_api_key():
                        next_page_token = None
                        continue
                    else:
                        self.logger.warning(
                            "No more API keys available; skipping further API search calls this run."
                        )
                        break
                self.logger.warning(
                    "YouTube Data API search failed; skipping further API search calls this run."
                )
                break

            for item in response.get("items", []):
                video_id = item["id"]["videoId"]
                snippet = item["snippet"]
                title = snippet["title"]
                upload_date_api = snippet.get("publishedAt")

                default_lang = snippet.get("defaultAudioLanguage") or snippet.get(
                    "defaultLanguage"
                )

                if video_id in seen_ids_set:
                    continue

                try:
                    details = self.youtube.videos().list(  # type: ignore[call-arg]
                        part="contentDetails",
                        id=video_id,
                    ).execute()
                except Exception as exc:  # noqa: BLE001
                    if "quotaExceeded" in str(exc):
                        self.logger.warning(
                            "YouTube videos().list quota exceeded for current key; attempting to switch API key."
                        )
                        if self._switch_to_next_api_key():
                            # retry this video with new key by not marking it seen
                            continue
                        else:
                            self.logger.warning(
                                "No more API keys available; stopping API discovery this run."
                            )
                            break
                    continue

                if not details.get("items"):
                    continue

                content_details = details["items"][0].get("contentDetails", {})
                iso_duration = content_details.get("duration", "P0D")
                duration_seconds = self._duration_to_seconds(iso_duration)

                if self._is_short_video(duration_seconds):
                    continue

                meta = self._fetch_video_metadata_pytubefix(video_id)
                if not meta.get("duration"):
                    meta["duration"] = duration_seconds

                if not self._contains_query_keyword(
                    title, meta.get("description") or ""
                ):
                    continue

                upload_date_meta = meta.get("uploadDate")
                final_upload_date = upload_date_api or upload_date_meta

                if not self._passes_date_filter(final_upload_date):
                    continue

                meta_language = meta.get("language")
                final_language = default_lang or meta_language

                channel_id = meta.get("channelId")
                if channel_id:
                    channel_avg_views = self._fetch_channel_average_views(channel_id)
                else:
                    channel_avg_views = None

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
            f"[API] Found {len(new_items)} new non-short relevant videos before shuffling."
        )
        random.shuffle(new_items)
        return new_items

    # -------------------------------------------------------------------------
    # Discovery via pytubefix Search
    # -------------------------------------------------------------------------
    def _discover_with_pytubefix_search(self, seen_ids_set: set[str],
                                        seen_ids_list: List[str]) -> List[Dict[str, Any]]:
        """Discover new relevant videos using pytubefix contrib Search."""
        if self._pytube_disabled:
            self.logger.info(
                "pytubefix has been disabled due to previous BotDetection; "
                "skipping Search discovery."
            )
            return []

        new_items: List[Dict[str, Any]] = []

        filters = (
            Filter.create()
            .type(Filter.Type.VIDEO)
            .sort_by(Filter.SortBy.RELEVANCE)
        )

        max_results = self.max_pages * self.results_per_page
        count = 0

        self.logger.info("Starting pytubefix Search discovery...")

        try:
            search_obj = Search(self.query, filters=filters)
        except pytube_exceptions.BotDetection:
            self.logger.warning(
                "pytubefix BotDetection when initializing Search; "
                "disabling pytubefix for the rest of this run"
            )
            self._pytube_disabled = True
            return []
        except Exception:
            self.logger.warning(
                "Failed to initialize pytubefix Search; skipping Search discovery this run"
            )
            return []

        for v in search_obj.videos:
            if self._pytube_disabled:
                break

            if count >= max_results:
                break
            count += 1

            video_id = getattr(v, "video_id", None) or getattr(v, "videoId", None)
            if not video_id:
                watch_url = getattr(v, "watch_url", "") or ""
                match = re.search(r"v=([0-9A-Za-z_-]{11})", watch_url)
                video_id = match.group(1) if match else None

            if not video_id or video_id in seen_ids_set:
                continue

            try:
                title = getattr(v, "title", "") or ""
            except pytube_exceptions.BotDetection:
                self.logger.warning(
                    f"pytubefix BotDetection when accessing title for video {video_id}; "
                    "disabling pytubefix Search for the rest of this run"
                )
                self._pytube_disabled = True
                break
            except Exception:
                self.logger.warning(
                    f"Failed to get title from pytubefix search result; skipping video {video_id}"
                )
                title = ""

            try:
                duration_seconds = int(getattr(v, "length", 0))
            except Exception:
                duration_seconds = 0

            meta = self._fetch_video_metadata_pytubefix(video_id)

            if not duration_seconds and meta.get("duration"):
                duration_seconds = meta["duration"]

            if duration_seconds and self._is_short_video(duration_seconds):
                continue

            effective_title = title or meta.get("title", "") or ""
            if not self._contains_query_keyword(
                effective_title, meta.get("description") or ""
            ):
                continue

            upload_date = meta.get("uploadDate")
            if not self._passes_date_filter(upload_date):
                continue

            channel_id = meta.get("channelId")
            if channel_id:
                channel_avg_views = self._fetch_channel_average_views(channel_id)
            else:
                channel_avg_views = None

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
                    "uploadDate": upload_date,
                    "language": meta.get("language"),
                    "channel_average_views": channel_avg_views,
                }
            )

        self.logger.info(
            f"[pytubefix Search] Found {len(new_items)} new non-short relevant videos before shuffling."
        )
        random.shuffle(new_items)
        return new_items

    # -------------------------------------------------------------------------
    # Load existing JSON (new structure only)
    # -------------------------------------------------------------------------
    def _load_existing_by_id(self) -> Dict[str, Dict[str, Any]]:
        """Load existing videos from JSON file into a dict keyed by videoId."""
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
        """Discover and enrich videos, then persist them to JSON."""
        existing_by_id = self._load_existing_by_id()
        existing_keys = set(existing_by_id.keys())

        seen_ids_list: List[str] = []
        seen_ids_set: set[str] = set(existing_keys)

        if os.path.exists(self.seen_file):
            with open(self.seen_file, "r", encoding="utf-8") as f:
                for line in f:
                    vid = line.strip()
                    if vid and vid not in seen_ids_set:
                        seen_ids_list.append(vid)
                        seen_ids_set.add(vid)

        for vid in existing_keys:
            if vid not in seen_ids_list:
                seen_ids_list.append(vid)

        all_new_items: List[Dict[str, Any]] = []

        if self.youtube is not None:
            api_items = self._discover_with_api(seen_ids_set, seen_ids_list)
            all_new_items.extend(api_items)

        pytube_items = self._discover_with_pytubefix_search(seen_ids_set, seen_ids_list)
        all_new_items.extend(pytube_items)

        self.logger.info(
            f"Total new non-short relevant videos discovered this run: {len(all_new_items)}"
        )

        combined_by_id: Dict[str, Dict[str, Any]] = {}

        for vid, meta in existing_by_id.items():
            combined_by_id[vid] = dict(meta) if isinstance(meta, dict) else {}

        for item in all_new_items:
            vid = item.get("videoId")
            if not vid:
                continue
            meta = combined_by_id.get(vid, {})
            for key, value in item.items():
                if key == "videoId":
                    continue
            # include new fields
                meta[key] = value
            combined_by_id[vid] = meta

        final_keys = set(combined_by_id.keys())
        unique_new_ids = final_keys - existing_keys
        self.logger.info(
            f"Ensuring metadata for {len(unique_new_ids)} newly added videos"
        )

        for vid in unique_new_ids:
            self._ensure_metadata_for_item(vid, combined_by_id[vid])

        with open(self.json_output, "w", encoding="utf-8") as f:
            json.dump(combined_by_id, f, indent=4, ensure_ascii=False)

        with open(self.seen_file, "w", encoding="utf-8") as f:
            for vid in seen_ids_list:
                f.write(vid + "\n")

        self.logger.info(
            f"Saved {len(unique_new_ids)} unique new entries to '{self.json_output}'. "
            f"Total entries stored: {len(combined_by_id)}"
        )
        self.logger.info(
            f"Total unique videos tracked (seen list size): {len(seen_ids_list)}"
        )

        return list(unique_new_ids)


# -------------------------------------------------------------------------
# Date window + API key loading helpers
# -------------------------------------------------------------------------
def _coerce_int(val: Any) -> Optional[int]:
    """Best-effort conversion to int; return None if not possible."""
    try:
        if val is None:
            return None
        if isinstance(val, int):
            return val
        s = str(val).strip()
        if not s:
            return None
        return int(s)
    except Exception:
        return None


def _parse_date_only(val: Any) -> Optional[datetime.date]:
    """
    Parse a config date value to a datetime.date.

    Accepts:
        - 'YYYY-MM-DD'
        - 'YYYY-MM-DDTHH:MM:SS'
        - 'YYYY-MM-DDTHH:MM:SSZ'
    """
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    if "T" in s:
        s = s.split("T", 1)[0]
    try:
        return datetime.date.fromisoformat(s)
    except ValueError:
        return None


def _add_months(d: datetime.date, months: int) -> datetime.date:
    """Add a number of months to a date, clamping the day to month length."""
    month_index = (d.month - 1) + months
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    day = min(d.day, last_day)
    return datetime.date(year, month, day)


def _compute_date_bounds_with_window() -> Tuple[Optional[str], Optional[str], bool]:
    """
    Compute (date_before_cfg, date_after_cfg, window_finished) to be passed
    into ASMRFetcher as (published_before, published_after).

    Behaviour:
        * If 'date_window_months' is not configured or <= 0:
            - Return raw 'date_before' (or legacy 'date') and 'date_after'.
            - window_finished = False.

        * If 'date_window_months' > 0 and both 'date_before' and 'date_after'
          are configured:
            - Interpret the two dates purely as bounds:
                earlier one  -> global_start
                later   one  -> global_end
            - Persist a progress file
              '<data_folder>/date_window_state.json' with:
                  {"next_start": "YYYY-MM-DD"}
            - For each run:
                current_start = max(global_start, next_start or global_start)
                if current_start >= global_end:
                    -> return (None, None, True)
                current_end   = min(current_start + window_months, global_end)
                save next_start = current_end

              Then:
                date_before_cfg = current_end   (published_before)
                date_after_cfg  = current_start (published_after)

        * Once current_start reaches or passes global_end, the function
          returns (None, None, True). The main block will then skip
          calling fetch_asmr_videos() (so we don't pass dates again).
    """
    raw_before = common.get_configs("date_before") or common.get_configs("date")
    raw_after = common.get_configs("date_after")
    raw_window = common.get_configs("date_window_months")

    window_months = _coerce_int(raw_window) or 0

    if window_months <= 0:
        return raw_before, raw_after, False

    if not raw_before or not raw_after:
        return raw_before, raw_after, False

    d_before = _parse_date_only(raw_before)
    d_after = _parse_date_only(raw_after)

    if not d_before or not d_after:
        return raw_before, raw_after, False

    global_start = min(d_before, d_after)
    global_end = max(d_before, d_after)

    if global_start == global_end:
        return None, None, True

    data_folder = common.get_configs("data") or "."
    os.makedirs(data_folder, exist_ok=True)
    state_path = os.path.join(data_folder, "date_window_state.json")

    next_start_date: Optional[datetime.date] = None
    if os.path.exists(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state_data = json.load(f)
            stored_start = state_data.get("next_start")
            parsed_start = _parse_date_only(stored_start)
            if parsed_start:
                next_start_date = parsed_start
        except Exception:
            next_start_date = None

    if next_start_date is None or next_start_date < global_start:
        current_start = global_start
    else:
        current_start = next_start_date

    if current_start >= global_end:
        return None, None, True

    candidate_end = _add_months(current_start, window_months)
    if candidate_end > global_end:
        current_end = global_end
    else:
        current_end = candidate_end

    try:
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({"next_start": current_end.isoformat()}, f, indent=2)
    except Exception:
        pass

    date_before_cfg = current_end.isoformat()
    date_after_cfg = current_start.isoformat()

    return date_before_cfg, date_after_cfg, False


def _load_api_keys_from_secrets() -> List[str]:
    """
    Load one or more API keys from secrets.

    Supports:
        - google-api-keys: list or comma/semicolon separated string
        - google-api-key: single key (fallback)
    """
    raw = common.get_secrets("google-api-keys") or common.get_secrets("google-api-key")
    keys: List[str] = []

    if not raw:
        return keys

    if isinstance(raw, str):
        parts = re.split(r"[;,]", raw)
        keys = [p.strip() for p in parts if p.strip()]
    elif isinstance(raw, (list, tuple, set)):
        for item in raw:
            s = str(item).strip()
            if s:
                keys.append(s)
    else:
        s = str(raw).strip()
        if s:
            keys.append(s)

    return keys


# -------------------------------------------------------------------------
# Standalone execution entry point (cron-friendly, multi-key)
# -------------------------------------------------------------------------
api_keys = _load_api_keys_from_secrets()

secret = SimpleNamespace(
    API_KEYS=api_keys,
)

date_before_cfg, date_after_cfg, window_finished = _compute_date_bounds_with_window()

fetcher = ASMRFetcher(
    api_keys=secret.API_KEYS,
    query=common.get_configs("query"),
    max_pages=100,
    results_per_page=50,
    seen_file="seen_video_ids.txt",
    json_output="asmr_results.json",
    published_before=date_before_cfg,
    published_after=date_after_cfg,
)

if __name__ == "__main__":
    raw_window = common.get_configs("date_window_months")
    window_months = _coerce_int(raw_window) or 0

    if window_months > 0 and window_finished:
        fetcher.logger.info(
            "Date windowing: configured range has already been fully processed. "
            "Skipping fetch_asmr_videos() for this run."
        )
    else:
        fetcher.logger.info(
            f"Running with published_after={date_after_cfg}, "
            f"published_before={date_before_cfg}"
        )
        fetcher.fetch_asmr_videos()
