# -*- coding: utf-8 -*-
"""Enrich existing YouTube JSON with uploadDate, language, and channel_average_views.

This script is intended to be run on an *existing* JSON file containing
YouTube video metadata keyed by videoId.

It will:
    * Load the JSON file.
    * For each video entry:
        - Add / fix `uploadDate` if missing or null.
        - Add / fix `language` using langdetect (fallback).
        - Add / fix `channel_average_views` using the YouTube Data API.
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
    * Only hit pytubefix / YouTube API when needed (missing fields).
    * Cache channel statistics per channel ID to avoid repeated calls.
    * Stop calling the channel stats API for this run if quotaExceeded occurs.
    * Play nicely with your CustomLogger (no dangerous `{}` in log messages).

Requirements:
    pip install pytubefix google-api-python-client langdetect

Configuration:
    * `common.get_configs("data")` provides the data folder.
    * JSON filename is configurable; defaults to "asmr_results.json".
    * Google API key is fetched via `common.get_secrets("google-api-key")`.

Author:
    Shadab Alam <shaadalam.5u@gmail.com>
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
    """Enrich existing JSON metadata with uploadDate, language, and channel_average_views.

    This class:
      * Loads a JSON file (dict keyed by videoId).
      * For each video:
          - Fetches uploadDate, language, and channel_average_views (when possible).
          - Updates only missing/empty fields to avoid clobbering existing values.
      * Writes the updated JSON file back to disk.

    Attributes:
        json_path: Full path to the JSON file to update.
        youtube: YouTube Data API client or None if no API key is set.
        logger: Custom logger instance for structured logging.
        _channel_stats_cache: In-memory cache for channel statistics lookups.
        _channel_stats_quota_exceeded: Flag set to True once quotaExceeded is
            encountered, to avoid further failing calls this run.
    """

    def __init__(
        self,
        api_key: Optional[str],
        json_path: str,
    ) -> None:
        """Initialize JSONMetadataEnricher.

        Args:
            api_key: YouTube API key. If None or empty, channel_average_views
                will not be populated (remains None).
            json_path: Path to the JSON file to enrich.
        """
        self.json_path = json_path
        self.api_key = api_key or None

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
                "No API key provided. channel_average_views will be left as None."
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

    def _is_missing(self, val: Any) -> bool:
        """Check whether a metadata value should be considered missing.

        Args:
            val: Value to check.

        Returns:
            True if the value is None or an empty/whitespace-only string;
            False otherwise.
        """
        if val is None:
            return True
        if isinstance(val, str) and val.strip() == "":
            return True
        return False

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

            # Cache result (including None) so we don't refetch this run.
            self._channel_stats_cache[channel_id] = avg
            return avg

        except Exception as exc:  # noqa: BLE001
            # Do NOT put exc string into the format message (it may contain braces).
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
    # pytubefix metadata helper
    # -------------------------------------------------------------------------
    def _fetch_video_metadata_pytubefix(self, video_id: str) -> Dict[str, Any]:
        """Fetch video metadata via pytubefix (for uploadDate, language, etc.).

        This method uses pytubefix's `YouTube` class to load metadata for a single
        video and returns a dictionary of fields that are useful for enrichment.

        Args:
            video_id: YouTube video ID (11-character string).

        Returns:
            A dictionary with keys:
                - "uploadDate" (str or None)
                - "language" (str or None)
                - "description" (str or None)
                - "title" (str or None)
                - "channelId" (str or None)
                - "author" (str or None)

            Any field not retrievable will be returned as None.
        """
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
            }

        title = getattr(yt, "title", None)
        description = getattr(yt, "description", None)
        publish_raw = getattr(yt, "publish_date", None)
        channel_id = getattr(yt, "channel_id", None)
        author = getattr(yt, "author", None)

        upload_date = self._normalize_upload_date(publish_raw)
        combined_text = f"{title or ''} {description or ''}"
        language = self._detect_language(combined_text)

        return {
            "uploadDate": upload_date,
            "language": language,
            "description": description,
            "title": title,
            "channelId": channel_id,
            "author": author,
        }

    # -------------------------------------------------------------------------
    # JSON load/save
    # -------------------------------------------------------------------------
    def _load_json(self) -> Dict[str, Dict[str, Any]]:
        """Load the JSON file into a dict keyed by videoId.

        Returns:
            Mapping from videoId to its metadata. If the JSON file does not
            exist or cannot be parsed, an empty dict is returned.
        """
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

        # Ensure each value is a dict.
        result: Dict[str, Dict[str, Any]] = {}
        for vid, meta in data.items():
            if not isinstance(meta, dict):
                meta = {}
            result[str(vid)] = meta

        return result

    def _save_json(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Save the updated JSON back to disk.

        Args:
            data: Mapping from videoId to metadata dictionaries.
        """
        tmp_path = self.json_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        # Replace original atomically (best-effort on most OSes).
        os.replace(tmp_path, self.json_path)
        self.logger.info("Updated JSON written to '{}'.".format(self.json_path))

    # -------------------------------------------------------------------------
    # Enrichment logic
    # -------------------------------------------------------------------------
    def _enrich_single_video(self, video_id: str, meta: Dict[str, Any]) -> None:
        """Enrich a single video's metadata in-place.

        This will:
            * Fill missing uploadDate using pytubefix.
            * Fill missing language using pytubefix and/or langdetect.
            * Fill missing channelId / author if possible.
            * Fill missing channel_average_views using YouTube Data API.

        Args:
            video_id: ID of the video to enrich.
            meta: Metadata dictionary to update in-place.
        """
        # Avoid unnecessary network calls: only fetch pytubefix metadata
        # if at least one of these fields is missing.
        need_pytube = any(
            self._is_missing(meta.get(field))
            for field in ("uploadDate", "language", "description", "title", "channelId", "author")
        )

        extra: Dict[str, Any] = {}
        if need_pytube:
            extra = self._fetch_video_metadata_pytubefix(video_id)

        # uploadDate
        if self._is_missing(meta.get("uploadDate")) and extra:
            meta["uploadDate"] = extra.get("uploadDate")

        # description
        if self._is_missing(meta.get("description")) and extra.get("description"):
            meta["description"] = extra.get("description")

        # title
        if self._is_missing(meta.get("title")) and extra.get("title"):
            meta["title"] = extra.get("title")

        # channelId
        if self._is_missing(meta.get("channelId")) and extra.get("channelId"):
            meta["channelId"] = extra.get("channelId")

        # author
        if self._is_missing(meta.get("author")) and extra.get("author"):
            meta["author"] = extra.get("author")

        # language from extra, or detect from title+description.
        if self._is_missing(meta.get("language")):
            language = None
            if extra:
                language = extra.get("language")
            if not language:
                combined_text = "{} {}".format(
                    meta.get("title", "") or "",
                    meta.get("description", "") or "",
                )
                language = self._detect_language(combined_text)
            meta["language"] = language

        # channel_average_views via API, if possible.
        if "channel_average_views" not in meta or meta.get("channel_average_views") is None:
            channel_id = meta.get("channelId")
            if channel_id:
                meta["channel_average_views"] = self._fetch_channel_average_views(
                    channel_id
                )

    def enrich_json(self) -> None:
        """Main entry point to enrich the JSON file.

        This method:
            1. Loads the JSON.
            2. Iterates over all video entries, enriching each one in-place.
            3. Writes the updated JSON back to disk.
        """
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
            except Exception as exc:  # noqa: F841
                self.logger.warning(
                    f"Failed to enrich video {video_id} at index {idx}; skipping this video."
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
    # Get API key (may be None/empty; then channel_average_views will stay None).
    secret_api = common.get_secrets("google-api-key")

    # Resolve data folder and JSON filename from configs.
    data_folder = common.get_configs("data")
    os.makedirs(data_folder, exist_ok=True)

    # You can change this to another filename if needed.
    json_filename = "asmr_results.json"
    json_path = os.path.join(data_folder, json_filename)

    enricher = JSONMetadataEnricher(
        api_key=secret_api,  # type: ignore
        json_path=json_path,
    )
    enricher.enrich_json()
