# -*- coding: utf-8 -*-
"""Enrich existing YouTube JSON with uploadDate, language, views, likes, and channel_average_views.

This script is intended to be run on an *existing* JSON file containing
YouTube video metadata keyed by videoId.

It will:
    * Load the JSON file.
    * Sync with a 'seen_videos_id.txt' file:
        - For IDs present in the txt but missing in JSON:
            * Try to fetch metadata and create new JSON entries.
            * If the video is no longer available, remove it from the txt
              (but never remove if the video already exists in JSON).
    * For each video entry:
        - Add / fix `uploadDate` if missing or null.
        - Add / fix `language` using langdetect (fallback).
        - Add / fix `channel_average_views` using the YouTube Data API.
        - Optionally refresh `views` and `likes` (API + pytubefix).
    * Write the updated JSON back to disk (by default in-place).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

from googleapiclient.discovery import build  # type: ignore
from langdetect import DetectorFactory, detect
from pytubefix import YouTube

from logmod import logs
import common
from custom_logger import CustomLogger

# Make langdetect deterministic (otherwise results can vary run-to-run).
DetectorFactory.seed = 0


class JSONMetadataEnricher:
    """Enrich existing JSON metadata with uploadDate, language, views, likes, and channel_average_views."""

    def __init__(self, api_key: Optional[str], json_path: str, update_views_likes: bool = False,
                 force_refresh_channel_avg: bool = False) -> None:
        """Initialize JSONMetadataEnricher."""
        self.json_path = json_path
        self.api_key = api_key or None
        self.update_views_likes = update_views_likes
        self.force_refresh_channel_avg = force_refresh_channel_avg

        # NEW: path to the seen_videos_id.txt file (same folder as JSON)
        self.seen_videos_path = os.path.join(
            os.path.dirname(self.json_path), "seen_videos_id.txt"
        )

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
        """Normalize numeric fields (views/likes/duration) to integers when possible."""
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

        By default, treat None, empty strings, and numeric 0 as missing.
        This makes sure things like duration=0, views=0, etc. will be updated.
        """
        if val is None:
            return True

        # numeric 0 (int/float) is treated as missing
        if isinstance(val, (int, float)) and val == 0:
            return True

        # empty or whitespace-only strings are missing
        if isinstance(val, str) and not val.strip():
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
        """Fetch video metadata via pytubefix (uploadDate, language, views, likes, duration, etc.)."""
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
                "duration": None,
            }

        title = getattr(yt, "title", None)
        description = getattr(yt, "description", None)
        publish_raw = getattr(yt, "publish_date", None)
        channel_id = getattr(yt, "channel_id", None)
        author = getattr(yt, "author", None)
        views_raw = getattr(yt, "views", None)
        likes_raw = getattr(yt, "likes", None)
        length_raw = getattr(yt, "length", None)

        upload_date = self._normalize_upload_date(publish_raw)
        combined_text = f"{title or ''} {description or ''}"
        language = self._detect_language(combined_text)
        views = self._normalize_int(views_raw)
        likes = self._normalize_int(likes_raw)
        duration = self._normalize_int(length_raw)

        return {
            "uploadDate": upload_date,
            "language": language,
            "description": description,
            "title": title,
            "channelId": channel_id,
            "author": author,
            "views": views,
            "likes": likes,
            "duration": duration,
        }

    # NEW: dedicated helper for *new* IDs from seen_videos_id.txt
    def _try_fetch_new_video_metadata(self, video_id: str) -> Tuple[Optional[Dict[str, Any]], bool]:
        """Fetch metadata for a new videoId, and detect if it's no longer available.

        Returns:
            (meta, unavailable_flag)
            meta: dict with metadata if successful, else None.
            unavailable_flag: True if we are confident the video is no longer available.
        """
        url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            yt = YouTube(url)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            # Heuristic: treat typical VideoUnavailable messages as "no longer available".
            unavailable = (
                "VideoUnavailable" in msg
                or "This video is unavailable" in msg
                or "This video is private" in msg
                or "404" in msg
            )
            if unavailable:
                self.logger.info(
                    "Video {} appears to be unavailable or private; will remove from seen_videos.".format(
                        video_id
                    )
                )
            else:
                self.logger.warning(
                    "Failed to fetch metadata for new video {} (transient error?); keeping it in seen_videos.".format(
                        video_id
                    )
                )
            return None, unavailable

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

        meta = {
            "uploadDate": upload_date,
            "language": language,
            "description": description,
            "title": title,
            "channelId": channel_id,
            "author": author,
            "views": views,
            "likes": likes,
            # duration will be fetched later if needed
        }
        return meta, False

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
    # NEW: Sync JSON with seen_videos_id.txt
    # -------------------------------------------------------------------------
    def _sync_with_seen_videos(self, data: Dict[str, Dict[str, Any]]) -> None:
        """Ensure JSON covers all IDs in seen_videos_id.txt, and drop truly unavailable new videos from txt.

        Rules:
            * If an ID is in seen_videos_id.txt but not in JSON:
                - Try to fetch metadata and add it to JSON.
                - If video is clearly unavailable, remove it from the txt.
                - If fetch fails for uncertain reasons, keep it in txt, don't add to JSON.
            * If an ID is already in JSON, never remove it from txt, even if unavailable.
        """
        if not os.path.exists(self.seen_videos_path):
            # Nothing to sync against.
            return

        try:
            with open(self.seen_videos_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "Could not read seen_videos file '{}': {}; skipping sync.".format(
                    self.seen_videos_path, exc
                )
            )
            return

        seen_ids_ordered = [line.strip() for line in lines if line.strip()]
        if not seen_ids_ordered:
            return

        json_ids = set(data.keys())
        ids_to_remove_from_seen = set()

        self.logger.info(
            "Syncing JSON with seen_videos_id.txt: {} IDs listed, {} JSON entries.".format(
                len(seen_ids_ordered), len(json_ids)
            )
        )

        for vid in seen_ids_ordered:
            if vid in json_ids:
                # Already represented in JSON, never delete from txt.
                continue

            # vid is new (present in txt but missing in json) -> try to fetch metadata
            meta, unavailable = self._try_fetch_new_video_metadata(vid)

            if unavailable:
                # Video appears gone; safe to drop from txt because it's not in JSON.
                ids_to_remove_from_seen.add(vid)
                continue

            if meta is None:
                # Some other error (network, etc.). Keep ID in txt, skip JSON insert.
                continue

            # Create a fresh metadata entry in JSON. channel_average_views will be
            # filled later (if allowed).
            data[vid] = {
                "title": meta.get("title"),
                "duration": meta.get("duration"),  # may be None; will be updated later if missing
                "channelId": meta.get("channelId"),
                "author": meta.get("author"),
                "views": meta.get("views"),
                "likes": meta.get("likes"),
                "description": meta.get("description"),
                "uploadDate": meta.get("uploadDate"),
                "language": meta.get("language"),
                "channel_average_views": None,
            }
            json_ids.add(vid)
            self.logger.info(
                "Added new video {} from seen_videos_id.txt into JSON.".format(vid)
            )

        # Rewrite seen_videos_id.txt excluding IDs we decided to remove.
        if ids_to_remove_from_seen:
            try:
                with open(self.seen_videos_path, "w", encoding="utf-8") as f:
                    for vid in seen_ids_ordered:
                        if vid not in ids_to_remove_from_seen:
                            f.write(vid + "\n")
                self.logger.info(
                    "Removed {} unavailable video IDs from '{}': {}".format(
                        len(ids_to_remove_from_seen),
                        self.seen_videos_path,
                        ", ".join(sorted(ids_to_remove_from_seen)),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning(
                    "Failed to update seen_videos file '{}': {}".format(
                        self.seen_videos_path, exc
                    )
                )

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
        # duration is considered missing if None, empty, or 0 (handled by _is_missing)
        missing_duration = self._is_missing(meta.get("duration"))

        # What needs work?
        need_base_update = any(
            [
                missing_uploadDate,
                missing_description,
                missing_title,
                missing_channelId,
                missing_author,
                missing_language,
                missing_duration,
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
            return

        # --- decide if we need pytubefix at all ---
        need_pytube_for_structure = any(
            [
                missing_uploadDate,
                missing_description,
                missing_title,
                missing_channelId,
                missing_author,
                missing_duration,
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

        # --- uploadDate / description / title / channelId / author / duration ---
        if missing_uploadDate and extra:
            meta["uploadDate"] = extra.get("uploadDate")

        if missing_description and extra.get("description") is not None:
            meta["description"] = extra.get("description")

        if missing_title and extra.get("title") is not None:
            meta["title"] = extra.get("title")

        if missing_channelId and extra.get("channelId") is not None:
            meta["channelId"] = extra.get("channelId")

        if missing_author and extra.get("author") is not None:
            meta["author"] = extra.get("author")

        if missing_duration and extra.get("duration") is not None:
            meta["duration"] = extra.get("duration")

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

        # NEW: sync JSON with seen_videos_id.txt before enrichment
        self._sync_with_seen_videos(data)

        total = len(data)
        self.logger.info(
            "Loaded {} video entries from '{}' after sync.".format(total, self.json_path)
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
