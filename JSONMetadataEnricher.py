# -*- coding: utf-8 -*-
"""Enrich existing YouTube JSON with uploadDate, language, views, likes, and channel_average_views.

This script is intended to be run on an *existing* JSON file containing
YouTube video metadata keyed by videoId.

It will:
    * Load the JSON file.
    * For each video entry:
        - Add / fix `uploadDate` if missing or null.
        - Add / fix `language` using langdetect (fallback).
        - Add / fix `channel_average_views` using the YouTube Data API.
        - Optionally refresh `views` and `likes` (API + pytubefix).
    * Write the updated JSON back to disk (by default in-place).

Expected JSON structure (input and output):

    {
        "<videoId>": {
            "title": "...",
            "duration": 1234,
            "channelId": "...",
            "author": "...",
            "views": 12345,
            "likes": 678,
            "description": "...",

            // New/updated fields:
            "uploadDate": "2025-01-01T12:34:56Z",
            "language": "en",
            "channel_average_views": 125000.0
        },
        ...
    }

Design goals:
    * Only hit pytubefix / YouTube API when needed (missing fields or refresh flags).
    * Skip updating any value that is already present unless the relevant flag is True.
    * Cache channel statistics per channel ID to avoid repeated calls.
    * Stop calling the channel stats API for this run if quotaExceeded occurs.
    * Play nicely with your CustomLogger (no dangerous `{}` in log messages).

Configuration:
    * `common.get_configs("data")` provides the data folder.
    * JSON filename is configurable; defaults to "asmr_results.json".
    * Google API key is fetched via `common.get_secrets("google-api-key")`.

Author:
    Shadab Alam <md_shadab_alam@outlook.com>
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from googleapiclient.discovery import build  # type: ignore
from langdetect import DetectorFactory, detect
from pytubefix import YouTube

from logmod import logs
import common
from custom_logger import CustomLogger

# Make langdetect deterministic (otherwise results can vary run-to-run).
DetectorFactory.seed = 0


class JSONMetadataEnricher:
    """Enrich existing JSON metadata with uploadDate, language, views, likes, and channel_average_views.

    This class:
      * Loads a JSON file (dict keyed by videoId).
      * For each video:
          - Fetches uploadDate, language, and channel_average_views (when possible).
          - Optionally refreshes views and likes.
          - Updates only missing/empty fields by default, unless flags say otherwise.
      * Writes the updated JSON file back to disk.

    Attributes:
        json_path: Full path to the JSON file to update.
        youtube: YouTube Data API client or None if no API key is set.
        logger: Custom logger instance for structured logging.
        update_views_likes: If True, refresh views/likes even if they are present.
        force_refresh_channel_avg: If True, recompute channel_average_views even if present.
        _channel_stats_cache: In-memory cache for channel statistics lookups.
        _channel_stats_quota_exceeded: Flag set to True once quotaExceeded is
            encountered, to avoid further failing calls this run.
    """

    def __init__(
        self,
        api_key: Optional[str],
        json_path: str,
        update_views_likes: bool = False,
        force_refresh_channel_avg: bool = False,
    ) -> None:
        """Initialize JSONMetadataEnricher.

        Args:
            api_key: YouTube API key. If None or empty, channel_average_views
                and API-based views/likes will not be populated.
            json_path: Path to the JSON file to enrich.
            update_views_likes: If True, update views/likes even when they exist.
            force_refresh_channel_avg: If True, recompute channel_average_views
                even when it exists.
        """
        self.json_path = json_path
        self.api_key = api_key or None
        self.update_views_likes = update_views_likes
        self.force_refresh_channel_avg = force_refresh_channel_avg

        # Initialize logging.
        logs(show_level=common.get_configs("logger_level"), show_color=True)
        self.logger = CustomLogger(__name__)

        # Initialize YouTube API client only if an API key is given.
        if self.api_key:
            self.youtube = build("youtube", "v3", developerKey=self.api_key)
            self.logger.info("YouTube Data API enabled (API key provided) for enrichment.")
        else:
            self.youtube = None
            self.logger.info(
                "No API key provided. channel_average_views and API-based views/likes will be left as None."
            )

        # In-memory cache to avoid repeated channel stats calls.
        # Key: channel_id, Value: channel_average_views (float or None).
        self._channel_stats_cache: Dict[str, Optional[float]] = {}

        # Flag: once quotaExceeded happens on channel stats, stop further calls this run.
        self._channel_stats_quota_exceeded: bool = False

    # -------------------------------------------------------------------------
    # Core helpers
    # -------------------------------------------------------------------------
    def _normalize_upload_date(self, val: Any) -> Optional[str]:
        """Normalize upload date to a string, if possible."""
        if val is None:
            return None

        try:
            import datetime

            if isinstance(val, (datetime.date, datetime.datetime)):
                return val.isoformat()
        except Exception:
            pass

        try:
            return str(val)
        except Exception:
            return None

    def _detect_language(self, text: str) -> Optional[str]:
        """Detect the language of the given text using langdetect."""
        text = (text or "").strip()
        if not text:
            return None
        try:
            return detect(text)
        except Exception:
            return None

    def _normalize_int(self, val: Any) -> Optional[int]:
        """Normalize numeric fields (views/likes) to integers when possible."""
        if val is None:
            return None
        if isinstance(val, int):
            return val
        if isinstance(val, float):
            return int(val)
        if isinstance(val, str):
            s = val.replace(",", "").strip()
            return int(s) if s.isdigit() else None
        return None

    def _is_missing(self, val: Any) -> bool:
        """Check whether a metadata value should be considered missing.

        For our purposes:

          * Missing if:
              - value is None, OR
              - value is an empty / whitespace-only string.
          * Present if:
              - value is any non-empty string, or
              - any non-None non-string (e.g., 0 for views).

        This ensures:
          - description is updated only when it's not present in a meaningful way
            (None, key absent, or blank).
          - description is NOT updated if it already has some text.
        """
        if val is None:
            return True

        if isinstance(val, str) and not val.strip():
            # "" or "   " -> treat as missing
            return True

        return False

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
            self.logger.warning(
                "Failed to fetch channel statistics; channel_average_views will be None for this channel"
            )

            if "quotaExceeded" in str(exc):
                self._channel_stats_quota_exceeded = True
                self.logger.warning(
                    "YouTube channel statistics quota exceeded; "
                    "channel_average_views will be skipped for the rest of this run"
                )

            self._channel_stats_cache[channel_id] = None
            return None

    # -------------------------------------------------------------------------
    # Video statistics helper (views/likes) via API
    # -------------------------------------------------------------------------
    def _fetch_video_stats_api(self, video_id: str) -> Dict[str, Optional[int]]:
        """Fetch video statistics (views, likes) via YouTube Data API."""
        if not self.youtube:
            return {"views": None, "likes": None}

        try:
            response = self.youtube.videos().list(  # type: ignore[call-arg]
                part="statistics",
                id=video_id,
            ).execute()
            items = response.get("items", [])
            if not items:
                return {"views": None, "likes": None}

            stats = items[0].get("statistics", {})
            view_count = self._normalize_int(stats.get("viewCount"))
            like_count = self._normalize_int(stats.get("likeCount"))
            return {"views": view_count, "likes": like_count}
        except Exception:
            self.logger.warning(
                "Failed to fetch video statistics via API for video {}; keeping existing views/likes".format(
                    video_id
                )
            )
            return {"views": None, "likes": None}

    # -------------------------------------------------------------------------
    # pytubefix metadata helper
    # -------------------------------------------------------------------------
    def _fetch_video_metadata_pytubefix(self, video_id: str) -> Dict[str, Any]:
        """Fetch video metadata via pytubefix (uploadDate, language, views, likes, etc.)."""
        url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            yt = YouTube(url)
        except Exception:  # noqa: BLE001
            self.logger.warning(
                "Failed to fetch pytubefix metadata for a video; using defaults for this video"
            )
            return {
                "uploadDate": None,
                "language": None,
                "description": None,
                "title": None,
                "channelId": None,
                "author": None,
                "views": None,
                "likes": None,
            }

        title = getattr(yt, "title", None)
        description = getattr(yt, "description", None)
        publish_raw = getattr(yt, "publish_date", None)
        channel_id = getattr(yt, "channel_id", None)
        author = getattr(yt, "author", None)
        views_raw = getattr(yt, "views", None)
        likes_raw = getattr(yt, "likes", None)

        upload_date = self._normalize_upload_date(publish_raw)
        combined_text = f"{title or ''} {description or ''}"
        language = self._detect_language(combined_text)
        views = self._normalize_int(views_raw)
        likes = self._normalize_int(likes_raw)

        return {
            "uploadDate": upload_date,
            "language": language,
            "description": description,
            "title": title,
            "channelId": channel_id,
            "author": author,
            "views": views,
            "likes": likes,
        }

    # -------------------------------------------------------------------------
    # JSON load/save
    # -------------------------------------------------------------------------
    def _load_json(self) -> Dict[str, Dict[str, Any]]:
        """Load the JSON file into a dict keyed by videoId."""
        if not os.path.exists(self.json_path):
            self.logger.warning(
                "JSON file '{}' does not exist. Nothing to enrich.".format(
                    self.json_path
                )
            )
            return {}

        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            self.logger.warning(
                "Existing JSON '{}' could not be parsed ({}); aborting.".format(
                    self.json_path, type(exc).__name__
                )
            )
            return {}

        if not isinstance(data, dict):
            self.logger.warning(
                "Existing JSON '{}' is not a dict; aborting.".format(self.json_path)
            )
            return {}

        result: Dict[str, Dict[str, Any]] = {}
        for vid, meta in data.items():
            if not isinstance(meta, dict):
                meta = {}
            result[str(vid)] = meta

        return result

    def _save_json(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Save the updated JSON back to disk."""
        tmp_path = self.json_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        os.replace(tmp_path, self.json_path)
        self.logger.info("Updated JSON written to '{}'.".format(self.json_path))

    # -------------------------------------------------------------------------
    # Enrichment logic
    # -------------------------------------------------------------------------
    def _enrich_single_video(self, video_id: str, meta: Dict[str, Any]) -> None:
        """Enrich a single video's metadata in-place."""
        # --- precompute "missing" status for all relevant fields ---
        missing_uploadDate = self._is_missing(meta.get("uploadDate"))
        missing_description = self._is_missing(meta.get("description"))
        missing_title = self._is_missing(meta.get("title"))
        missing_channelId = self._is_missing(meta.get("channelId"))
        missing_author = self._is_missing(meta.get("author"))
        missing_language = self._is_missing(meta.get("language"))
        missing_views = self._is_missing(meta.get("views"))
        missing_likes = self._is_missing(meta.get("likes"))
        missing_channel_avg = self._is_missing(meta.get("channel_average_views"))

        # What needs work?
        need_base_update = any(
            [
                missing_uploadDate,
                missing_description,
                missing_title,
                missing_channelId,
                missing_author,
                missing_language,
            ]
        )
        need_views_update = self.update_views_likes or missing_views
        need_likes_update = self.update_views_likes or missing_likes
        need_channel_avg_update = self.force_refresh_channel_avg or missing_channel_avg

        # --- fast path: if nothing needs to be updated, skip this video entirely ---
        if not (
            need_base_update
            or need_views_update
            or need_likes_update
            or need_channel_avg_update
        ):
            # All relevant values are present and no refresh flags are set.
            # Do NOT touch this entry; just move to the next one.
            return

        # --- decide if we need pytubefix at all ---
        # We only hit pytubefix when we *really* need extra metadata:
        #   - structural fields (uploadDate, description, title, channelId, author), or
        #   - language when we have no text to detect from, or
        #   - as a fallback for views/likes when the API gives nothing.
        need_pytube_for_structure = any(
            [
                missing_uploadDate,
                missing_description,
                missing_title,
                missing_channelId,
                missing_author,
            ]
        )
        # For language, only require pytube if language is missing AND we don't
        # already have any text (title/description) to run langdetect on.
        need_pytube_for_language = missing_language and (
            missing_title or missing_description
        )

        # For stats, we'll prefer the API first; pytube is only a fallback.
        need_stats = need_views_update or need_likes_update
        need_pytube = need_pytube_for_structure or need_pytube_for_language

        extra: Dict[str, Any] = {}
        if need_pytube:
            extra = self._fetch_video_metadata_pytubefix(video_id)

        # --- uploadDate / description / title / channelId / author ---
        if missing_uploadDate and extra:
            meta["uploadDate"] = extra.get("uploadDate")

        # Only update description when it's missing; never overwrite an existing
        # non-empty description.
        if missing_description and extra.get("description") is not None:
            meta["description"] = extra.get("description")

        if missing_title and extra.get("title") is not None:
            meta["title"] = extra.get("title")

        if missing_channelId and extra.get("channelId") is not None:
            meta["channelId"] = extra.get("channelId")

        if missing_author and extra.get("author") is not None:
            meta["author"] = extra.get("author")

        # --- language: from pytube if available, otherwise langdetect ---
        if missing_language:
            language = None
            if extra:
                language = extra.get("language")

            if not language:
                combined_text = "{} {}".format(
                    meta.get("title", "") or "",
                    meta.get("description", "") or "",
                ).strip()
                if combined_text:
                    language = self._detect_language(combined_text)

            meta["language"] = language

        # --- views & likes: API preferred, pytubefix fallback; never overwrite unless flag says so ---
        stats: Dict[str, Optional[int]] = {"views": None, "likes": None}
        if need_stats:
            stats = self._fetch_video_stats_api(video_id)

        if need_views_update:
            new_views: Optional[int] = stats.get("views")
            if new_views is None and extra:
                # Only use pytubefix if API gave us nothing.
                new_views = extra.get("views")
            if new_views is not None:
                meta["views"] = new_views

        if need_likes_update:
            new_likes: Optional[int] = stats.get("likes")
            if new_likes is None and extra:
                new_likes = extra.get("likes")
            if new_likes is not None:
                meta["likes"] = new_likes

        # --- channel_average_views: only if missing or force_refresh_channel_avg is True ---
        channel_id = meta.get("channelId")
        if self.force_refresh_channel_avg:
            if channel_id:
                meta["channel_average_views"] = self._fetch_channel_average_views(
                    channel_id
                )
        else:
            if missing_channel_avg and channel_id:
                meta["channel_average_views"] = self._fetch_channel_average_views(
                    channel_id
                )

    def enrich_json(self) -> None:
        """Main entry point to enrich the JSON file."""
        data = self._load_json()
        if not data:
            self.logger.info("No data loaded. Exiting without changes.")
            return

        total = len(data)
        self.logger.info(
            "Loaded {} video entries from '{}'.".format(total, self.json_path)
        )

        for idx, (video_id, meta) in enumerate(data.items(), start=1):
            try:
                self._enrich_single_video(video_id, meta)
            except Exception:
                self.logger.warning(
                    "Failed to enrich video {} at index {}; skipping this video.".format(
                        video_id, idx
                    )
                )

            if idx % 50 == 0 or idx == total:
                self.logger.info("Progress: {} / {} videos enriched.".format(idx, total))

        self._save_json(data)
        self.logger.info(
            "Enrichment complete. Total videos processed: {}".format(total)
        )


# -------------------------------------------------------------------------
# Standalone execution entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Get API key (may be None/empty; then channel_average_views & API stats may stay None).
    secret_api = common.get_secrets("google-api-key")

    # Resolve data folder and JSON filename from configs.
    data_folder = common.get_configs("data")
    os.makedirs(data_folder, exist_ok=True)

    json_filename = "asmr_results.json"
    json_path = os.path.join(data_folder, json_filename)

    enricher = JSONMetadataEnricher(
        api_key=secret_api,  # type: ignore
        json_path=json_path,
        update_views_likes=False,         # set True to refresh views/likes even when present
        force_refresh_channel_avg=False,  # set True to recompute channel_average_views even when present
    )
    enricher.enrich_json()
