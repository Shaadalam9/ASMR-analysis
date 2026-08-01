# -*- coding: utf-8 -*-
"""Recover YouTube JSON metadata from seen video ID text files.

This script is separate from main.py. It reads video IDs from one or more
seen*.txt files, compares them with the existing JSON, fetches missing metadata,
and writes the recovered data back to the original JSON file safely.

Important behaviour:
    * Input/output JSON: <data folder>/asmr_results.json
    * Existing JSON entries are kept.
    * Missing JSON entries are recovered from the YouTube Data API first.
    * pytubefix is used only as a fallback.
    * If a missing ID is confidently private, deleted, or unavailable, it is
      removed from the seen text files instead of being added as an empty JSON
      entry.
    * YouTube Data API quota exhaustion never causes IDs to be removed.
    * pytubefix HTTP 429 never causes IDs to be removed.
    * The JSON is written atomically and the previous JSON is backed up first.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from googleapiclient.discovery import build  # type: ignore
from langdetect import DetectorFactory, detect
from pytubefix import YouTube, exceptions as pytube_exceptions

from logmod import logs
import common
from custom_logger import CustomLogger


DetectorFactory.seed = 0


# -----------------------------------------------------------------------------
# User editable settings
# -----------------------------------------------------------------------------
JSON_FILENAME = "asmr_results.json"

# Write directly back to asmr_results.json. A timestamped backup is created first.
WRITE_IN_PLACE = True

# Recover IDs that are present in seen text files but missing from JSON.
RECOVER_IDS_MISSING_FROM_JSON = True

# Also fill missing fields in entries that already exist in JSON.
FILL_MISSING_FIELDS_IN_EXISTING_JSON = True

# If True, API values can replace existing views and likes. If False, views and
# likes are only filled when missing.
REFRESH_EXISTING_VIEWS_LIKES = False

# Your requested behaviour: if a missing ID is confirmed private/deleted/removed,
# remove it from the seen text files instead of creating an empty JSON entry.
REMOVE_CONFIRMED_UNAVAILABLE_IDS_FROM_SEEN_TXT = True
ADD_PLACEHOLDER_FOR_UNAVAILABLE_IDS = False

# YouTube Data API accepts up to 50 video IDs per videos().list call.
VIDEO_BATCH_SIZE = 50
CHANNEL_BATCH_SIZE = 50

# Save a checkpoint beside the JSON every N video batches.
CHECKPOINT_EVERY_N_BATCHES = 10

# Optional pause between API batches.
SLEEP_SECONDS_BETWEEN_BATCHES = 0.0

# Stop using pytubefix after the first HTTP 429. This avoids repeated web
# scraping requests and prevents 429 from being mistaken for private/deleted.
DISABLE_PYTUBEFIX_AFTER_HTTP_429 = True


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def _data_folder() -> str:
    folder = common.get_configs("data")
    if not folder:
        folder = "."
    os.makedirs(folder, exist_ok=True)
    return folder


def _now_stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _is_missing(value: Any, *, zero_is_missing: bool = False) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if zero_is_missing and isinstance(value, (int, float)) and value == 0:
        return True
    return False


def _normalise_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.replace(",", "").strip()
        return int(s) if s.isdigit() else None
    return None


def _normalise_upload_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    try:
        return str(value)
    except Exception:
        return None


def _detect_language(text: str) -> Optional[str]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return detect(text)
    except Exception:
        return None


def _duration_to_seconds(iso_duration: Optional[str]) -> Optional[int]:
    if not iso_duration:
        return None
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

    return None


def _empty_metadata(status: str = "placeholder") -> Dict[str, Any]:
    return {
        "title": None,
        "duration": None,
        "channelId": None,
        "author": None,
        "views": None,
        "likes": None,
        "description": None,
        "uploadDate": None,
        "language": None,
        "languageSource": None,
        "channel_average_views": None,
        "metadataCollectedAt": None,
        "recovery_status": status,
    }


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc)
    return "429" in message or "Too Many Requests" in message


def _looks_unavailable_error(exc: Exception) -> bool:
    message = str(exc)
    tokens = [
        "VideoUnavailable",
        "This video is unavailable",
        "This video is private",
        "This video has been removed",
        "Video has been removed",
        "Private video",
        "not available",
        "404",
    ]
    return any(token in message for token in tokens)


def _merge_metadata(
    target: Dict[str, Any],
    incoming: Dict[str, Any],
    *,
    overwrite_views_likes: bool = False,
    overwrite_all: bool = False,
) -> bool:
    changed = False

    for key, value in incoming.items():
        if key == "videoId" or value is None:
            continue

        zero_missing = key in {"duration"}
        current_missing = _is_missing(target.get(key), zero_is_missing=zero_missing)

        if overwrite_all:
            if target.get(key) != value:
                target[key] = value
                changed = True
            continue

        if key in {"views", "likes"}:
            if overwrite_views_likes or current_missing:
                if target.get(key) != value:
                    target[key] = value
                    changed = True
            continue

        if current_missing:
            if target.get(key) != value:
                target[key] = value
                changed = True

    return changed


# -----------------------------------------------------------------------------
# File helpers
# -----------------------------------------------------------------------------
def _extract_video_ids_from_line(line: str) -> List[str]:
    line = line.strip()
    if not line:
        return []

    ids: List[str] = []
    patterns = [
        r"(?:v=)([0-9A-Za-z_-]{11})",
        r"(?:youtu\.be/)([0-9A-Za-z_-]{11})",
        r"(?:shorts/)([0-9A-Za-z_-]{11})",
        r"(?:embed/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        ids.extend(re.findall(pattern, line))

    if not ids:
        ids.extend(re.findall(r"(?<![0-9A-Za-z_-])([0-9A-Za-z_-]{11})(?![0-9A-Za-z_-])", line))

    unique: List[str] = []
    seen = set()
    for video_id in ids:
        if video_id not in seen:
            seen.add(video_id)
            unique.append(video_id)
    return unique


def _discover_seen_files(folder: str) -> List[str]:
    candidate_names = {
        "seen_video_ids.txt",
        "seen_videos_id.txt",
    }

    try:
        for name in os.listdir(folder):
            lower = name.lower()
            if lower.endswith(".txt") and "seen" in lower and "video" in lower:
                candidate_names.add(name)
    except FileNotFoundError:
        return []

    paths = []
    for name in sorted(candidate_names):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            paths.append(path)
    return paths


def _load_seen_ids(folder: str) -> List[str]:
    paths = _discover_seen_files(folder)
    if not paths:
        logger.warning("No seen video text files found in '{}'.".format(folder))
        return []

    ordered: List[str] = []
    seen = set()

    for path in paths:
        count_before = len(ordered)
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    for video_id in _extract_video_ids_from_line(line):
                        if video_id not in seen:
                            seen.add(video_id)
                            ordered.append(video_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read '{}': {}".format(path, exc))
            continue

        logger.info(
            "Loaded {} unique new IDs from '{}'.".format(
                len(ordered) - count_before,
                path,
            )
        )

    logger.info("Total unique seen IDs loaded from text files: {}".format(len(ordered)))
    return ordered


def _remove_ids_from_seen_files(folder: str, ids_to_remove: Sequence[str]) -> int:
    remove_set = set(ids_to_remove)
    if not remove_set:
        return 0

    total_removed = 0
    for path in _discover_seen_files(folder):
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read '{}' for cleanup: {}".format(path, exc))
            continue

        changed = False
        new_lines: List[str] = []
        removed_from_file = 0

        for line in lines:
            ids_in_line = _extract_video_ids_from_line(line)
            if not ids_in_line:
                new_lines.append(line)
                continue

            ids_removed = [video_id for video_id in ids_in_line if video_id in remove_set]
            if not ids_removed:
                new_lines.append(line)
                continue

            changed = True
            removed_from_file += len(ids_removed)
            ids_to_keep = [video_id for video_id in ids_in_line if video_id not in remove_set]

            # Usually there is one ID per line. If a line has multiple IDs, keep
            # the remaining valid IDs one per line rather than deleting them.
            for video_id in ids_to_keep:
                new_lines.append(video_id + "\n")

        if not changed:
            continue

        backup_path = path + ".bak_removed_" + _now_stamp()
        tmp_path = path + ".tmp_" + _now_stamp()
        try:
            shutil.copy2(path, backup_path)
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            os.replace(tmp_path, path)
            total_removed += removed_from_file
            logger.info(
                "Removed {} confirmed unavailable ID occurrence(s) from '{}'. Backup: '{}'".format(
                    removed_from_file,
                    path,
                    backup_path,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to update seen file '{}': {}".format(path, exc))
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    return total_removed


def _load_json(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        logger.warning("Input JSON '{}' does not exist. Starting from empty JSON.".format(path))
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not parse '{}': {}. Refusing to overwrite the original JSON.".format(
                path,
                exc,
            )
        )

    if not isinstance(data, dict):
        raise RuntimeError("Input JSON '{}' is not a dict. Refusing to overwrite it.".format(path))

    cleaned: Dict[str, Dict[str, Any]] = {}
    for video_id, meta in data.items():
        if isinstance(meta, dict):
            cleaned[str(video_id)] = meta
        else:
            cleaned[str(video_id)] = {}

    logger.info("Loaded {} entries from '{}'.".format(len(cleaned), path))
    return cleaned


def _atomic_save_json(path: str, data: Dict[str, Dict[str, Any]]) -> None:
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)

    if os.path.exists(path):
        backup_path = path + ".bak_" + _now_stamp()
        shutil.copy2(path, backup_path)
        logger.info("Backed up existing JSON to '{}'.".format(backup_path))

    tmp_path = path + ".tmp_" + _now_stamp()
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    os.replace(tmp_path, path)
    logger.info("Saved {} entries to '{}'.".format(len(data), path))


def _save_checkpoint(path: str, data: Dict[str, Dict[str, Any]]) -> None:
    tmp_path = path + ".checkpoint.tmp"
    checkpoint_path = path + ".checkpoint"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.replace(tmp_path, checkpoint_path)
    logger.info("Checkpoint saved to '{}'.".format(checkpoint_path))


# -----------------------------------------------------------------------------
# Secrets and API helpers
# -----------------------------------------------------------------------------
def _load_api_keys_from_secrets() -> List[str]:
    raw = (
        os.environ.get("YOUTUBE_API_KEYS")
        or os.environ.get("YOUTUBE_API_KEY")
        or common.get_secrets("google-api-keys")
        or common.get_secrets("google-api-key")
    )
    keys: List[str] = []

    if not raw:
        return keys

    if isinstance(raw, str):
        parts = re.split(r"[;,]", raw)
        keys = [part.strip() for part in parts if part.strip()]
    elif isinstance(raw, (list, tuple, set)):
        for item in raw:
            s = str(item).strip()
            if s:
                keys.append(s)
    else:
        s = str(raw).strip()
        if s:
            keys.append(s)

    deduped: List[str] = []
    seen = set()
    for key in keys:
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


class YouTubeAPIClient:
    """Small wrapper around the YouTube Data API with key rotation."""

    def __init__(self, api_keys: Sequence[str]) -> None:
        self.api_keys = [key for key in api_keys if key]
        self.index = 0
        self.youtube = None
        self.quota_exhausted = False
        self._init_current_key()

    def _init_current_key(self) -> None:
        if not self.api_keys:
            self.youtube = None
            logger.info("No YouTube Data API keys found. Using pytubefix fallback only.")
            return

        key = self.api_keys[self.index]
        try:
            self.youtube = build(
                "youtube",
                "v3",
                developerKey=key,
                cache_discovery=False,
            )
            logger.info(
                "YouTube Data API enabled. Using key {}/{}.".format(
                    self.index + 1,
                    len(self.api_keys),
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to initialise YouTube API key {}/{}: {}".format(
                    self.index + 1,
                    len(self.api_keys),
                    exc,
                )
            )
            self._switch_key()

    def _switch_key(self) -> bool:
        self.index += 1
        if self.index >= len(self.api_keys):
            self.youtube = None
            self.quota_exhausted = True
            logger.warning("All YouTube Data API keys exhausted or failed.")
            return False
        self._init_current_key()
        return self.youtube is not None

    def execute(self, request_factory, context: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """Execute a YouTube API request.

        Returns:
            (response, status)
            status is one of: success, quota, error, no_client
        """
        if self.youtube is None:
            return None, "no_client"

        while self.youtube is not None:
            try:
                return request_factory(self.youtube).execute(), "success"
            except Exception as exc:  # noqa: BLE001
                message = str(exc)
                if "quotaExceeded" in message or "dailyLimitExceeded" in message:
                    logger.warning("Quota exceeded during {}; switching API key.".format(context))
                    if self._switch_key():
                        continue
                    return None, "quota"

                logger.warning("YouTube API request failed during {}: {}".format(context, exc))
                return None, "error"

        return None, "quota" if self.quota_exhausted else "no_client"


# -----------------------------------------------------------------------------
# Metadata fetchers
# -----------------------------------------------------------------------------
def _metadata_from_api_item(item: Dict[str, Any]) -> Dict[str, Any]:
    snippet = item.get("snippet", {}) or {}
    statistics = item.get("statistics", {}) or {}
    content_details = item.get("contentDetails", {}) or {}

    title = snippet.get("title")
    description = snippet.get("description")
    default_audio_language = snippet.get("defaultAudioLanguage")
    default_language = snippet.get("defaultLanguage")

    language = default_audio_language or default_language
    if not language:
        language = _detect_language("{} {}".format(title or "", description or ""))

    if default_audio_language:
        language_source = "youtube:defaultAudioLanguage"
    elif default_language:
        language_source = "youtube:defaultLanguage"
    elif language:
        language_source = "langdetect:title_description"
    else:
        language_source = None

    return {
        "title": title,
        "duration": _duration_to_seconds(content_details.get("duration")),
        "channelId": snippet.get("channelId"),
        "author": snippet.get("channelTitle"),
        "views": _normalise_int(statistics.get("viewCount")),
        "likes": _normalise_int(statistics.get("likeCount")),
        "description": description,
        "uploadDate": snippet.get("publishedAt"),
        "language": language,
        "languageSource": language_source,
        "channel_average_views": None,
        "metadataCollectedAt": _now_stamp(),
        "recovery_status": "recovered_from_api",
    }


def _fetch_video_metadata_api(
    api_client: YouTubeAPIClient,
    video_ids: Sequence[str],
) -> Tuple[Dict[str, Dict[str, Any]], List[str], str]:
    """Fetch video metadata from the YouTube Data API.

    Returns:
        (recovered_by_id, confirmed_unavailable_ids, status)

    confirmed_unavailable_ids is only populated when the API request succeeds.
    It remains empty on quota exhaustion or transient API errors.
    """
    if api_client.youtube is None or not video_ids:
        return {}, [], "no_client"

    response, status = api_client.execute(
        lambda youtube: youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=",".join(video_ids),
            maxResults=len(video_ids),
        ),
        context="videos().list",
    )
    if response is None or status != "success":
        return {}, [], status

    recovered: Dict[str, Dict[str, Any]] = {}
    for item in response.get("items", []):
        video_id = item.get("id")
        if video_id:
            recovered[str(video_id)] = _metadata_from_api_item(item)

    confirmed_unavailable = [video_id for video_id in video_ids if video_id not in recovered]
    return recovered, confirmed_unavailable, "success"


def _fetch_channel_average_views_api(
    api_client: YouTubeAPIClient,
    channel_ids: Sequence[str],
) -> Dict[str, Optional[float]]:
    if api_client.youtube is None or not channel_ids:
        return {}

    response, status = api_client.execute(
        lambda youtube: youtube.channels().list(
            part="statistics",
            id=",".join(channel_ids),
            maxResults=len(channel_ids),
        ),
        context="channels().list",
    )
    if response is None or status != "success":
        return {}

    result: Dict[str, Optional[float]] = {}
    for item in response.get("items", []):
        channel_id = item.get("id")
        stats = item.get("statistics", {}) or {}
        if not channel_id:
            continue

        total_views = _normalise_int(stats.get("viewCount")) or 0
        total_videos = _normalise_int(stats.get("videoCount")) or 0
        if total_videos > 0:
            result[str(channel_id)] = total_views / float(total_videos)
        else:
            result[str(channel_id)] = None

    return result


class PytubefixFallback:
    def __init__(self) -> None:
        self.disabled_by_429 = False
        self.logged_disabled = False

    def fetch(self, video_id: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """Fetch metadata with pytubefix.

        Returns:
            (metadata, status)
            status is one of: success, unavailable, rate_limited, bot_detection, error, disabled
        """
        if self.disabled_by_429:
            if not self.logged_disabled:
                logger.warning("pytubefix fallback is disabled for this run after HTTP 429.")
                self.logged_disabled = True
            return None, "disabled"

        url = "https://www.youtube.com/watch?v={}".format(video_id)

        try:
            try:
                yt = YouTube(url, "WEB")
            except TypeError:
                yt = YouTube(url)
        except pytube_exceptions.BotDetection:
            logger.warning("pytubefix bot detection hit for video {}. Keeping ID in seen files.".format(video_id))
            return None, "bot_detection"
        except Exception as exc:  # noqa: BLE001
            if _is_rate_limit_error(exc):
                logger.warning(
                    "pytubefix HTTP 429 for {}: {}. Keeping IDs in seen files.".format(
                        video_id,
                        exc,
                    )
                )
                if DISABLE_PYTUBEFIX_AFTER_HTTP_429:
                    self.disabled_by_429 = True
                    logger.warning("Disabling pytubefix fallback for the rest of this run due to HTTP 429.")
                return None, "rate_limited"
            if _looks_unavailable_error(exc):
                return None, "unavailable"
            logger.warning("pytubefix failed for {}: {}. Keeping ID in seen files.".format(video_id, exc))
            return None, "error"

        try:
            title = getattr(yt, "title", None)
            description = getattr(yt, "description", None)
            publish_raw = getattr(yt, "publish_date", None)
            channel_id = getattr(yt, "channel_id", None)
            author = getattr(yt, "author", None)
            views = _normalise_int(getattr(yt, "views", None))
            likes = _normalise_int(getattr(yt, "likes", None))

            try:
                duration = _normalise_int(getattr(yt, "length", None))
            except Exception:
                duration = None

            language = _detect_language("{} {}".format(title or "", description or ""))

            return {
                "title": title,
                "duration": duration,
                "channelId": channel_id,
                "author": author,
                "views": views,
                "likes": likes,
                "description": description,
                "uploadDate": _normalise_upload_date(publish_raw),
                "language": language,
                "channel_average_views": None,
                "recovery_status": "recovered_from_pytubefix",
            }, "success"
        except pytube_exceptions.BotDetection:
            logger.warning("pytubefix bot detection while reading video {}. Keeping ID in seen files.".format(video_id))
            return None, "bot_detection"
        except Exception as exc:  # noqa: BLE001
            if _is_rate_limit_error(exc):
                logger.warning(
                    "pytubefix metadata extraction HTTP 429 for {}: {}. Keeping IDs in seen files.".format(
                        video_id,
                        exc,
                    )
                )
                if DISABLE_PYTUBEFIX_AFTER_HTTP_429:
                    self.disabled_by_429 = True
                    logger.warning("Disabling pytubefix fallback for the rest of this run due to HTTP 429.")
                return None, "rate_limited"
            if _looks_unavailable_error(exc):
                return None, "unavailable"
            logger.warning("pytubefix metadata extraction failed for {}: {}. Keeping ID in seen files.".format(video_id, exc))
            return None, "error"


# -----------------------------------------------------------------------------
# Recovery logic
# -----------------------------------------------------------------------------
def _batched(items: Sequence[str], batch_size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start:start + batch_size])


def _needs_metadata_recovery(meta: Dict[str, Any]) -> bool:
    checks = [
        _is_missing(meta.get("title")),
        _is_missing(meta.get("duration"), zero_is_missing=True),
        _is_missing(meta.get("channelId")),
        _is_missing(meta.get("author")),
        _is_missing(meta.get("views")),
        _is_missing(meta.get("description")),
        _is_missing(meta.get("uploadDate")),
        _is_missing(meta.get("language")),
    ]
    return any(checks)


def _apply_channel_average_views(
    data: Dict[str, Dict[str, Any]],
    api_client: YouTubeAPIClient,
) -> None:
    if api_client.youtube is None:
        logger.info("Skipping channel_average_views because no YouTube Data API client is available.")
        return

    channel_ids: List[str] = []
    seen = set()

    for meta in data.values():
        channel_id = meta.get("channelId")
        if not channel_id:
            continue
        if not _is_missing(meta.get("channel_average_views")):
            continue
        if channel_id not in seen:
            seen.add(channel_id)
            channel_ids.append(channel_id)

    if not channel_ids:
        logger.info("No missing channel_average_views values to recover.")
        return

    logger.info("Recovering channel_average_views for {} channels.".format(len(channel_ids)))

    recovered_count = 0
    for batch in _batched(channel_ids, CHANNEL_BATCH_SIZE):
        averages = _fetch_channel_average_views_api(api_client, batch)
        if not averages:
            if api_client.quota_exhausted:
                logger.warning("Stopping channel_average_views recovery because API quota is exhausted.")
                break
            continue

        for meta in data.values():
            channel_id = meta.get("channelId")
            if channel_id in averages and _is_missing(meta.get("channel_average_views")):
                meta["channel_average_views"] = averages[channel_id]
                if averages[channel_id] is not None:
                    recovered_count += 1

    logger.info("Recovered channel_average_views for {} video entries.".format(recovered_count))


def recover_json_from_seen_ids() -> None:
    folder = _data_folder()
    json_path = os.path.join(folder, JSON_FILENAME)
    output_json_path = json_path

    api_keys = _load_api_keys_from_secrets()
    api_client = YouTubeAPIClient(api_keys)
    pytube = PytubefixFallback()

    seen_ids = _load_seen_ids(folder)
    if not seen_ids:
        logger.warning("No seen IDs found. Nothing to recover.")
        return

    data = _load_json(json_path)
    original_json_count = len(data)

    ids_missing_from_json = [video_id for video_id in seen_ids if video_id not in data]
    ids_existing_but_incomplete = [
        video_id
        for video_id in seen_ids
        if video_id in data and FILL_MISSING_FIELDS_IN_EXISTING_JSON and _needs_metadata_recovery(data[video_id])
    ]

    if not RECOVER_IDS_MISSING_FROM_JSON:
        ids_missing_from_json = []

    ids_missing_from_json_set = set(ids_missing_from_json)
    ids_to_recover = ids_missing_from_json + ids_existing_but_incomplete

    logger.info("Existing JSON entries: {}".format(original_json_count))
    logger.info("IDs in text files but missing from JSON: {}".format(len(ids_missing_from_json)))
    logger.info("Existing JSON entries with missing fields: {}".format(len(ids_existing_but_incomplete)))
    logger.info("Total IDs selected for recovery: {}".format(len(ids_to_recover)))

    api_recovered_count = 0
    pytube_recovered_count = 0
    placeholder_count = 0
    api_confirmed_unavailable_count = 0
    pytube_confirmed_unavailable_count = 0

    confirmed_unavailable_to_remove: List[str] = []
    confirmed_unavailable_seen = set()

    total_batches = (len(ids_to_recover) + VIDEO_BATCH_SIZE - 1) // VIDEO_BATCH_SIZE

    for batch_index, batch in enumerate(_batched(ids_to_recover, VIDEO_BATCH_SIZE), start=1):
        logger.info("Processing video batch {}/{} ({} IDs).".format(batch_index, total_batches, len(batch)))

        recovered_by_api, unavailable_by_api, api_status = _fetch_video_metadata_api(api_client, batch)
        api_recovered_count += len(recovered_by_api)

        for video_id, metadata in recovered_by_api.items():
            if video_id not in data:
                data[video_id] = _empty_metadata(status="created_before_api_merge")
            _merge_metadata(
                data[video_id],
                metadata,
                overwrite_views_likes=REFRESH_EXISTING_VIEWS_LIKES,
            )

        unavailable_by_api_set = set(unavailable_by_api)
        not_recovered_by_api = [video_id for video_id in batch if video_id not in recovered_by_api]

        for video_id in not_recovered_by_api:
            # Only remove if the API request succeeded and omitted the ID.
            # Do not remove anything when quota is exhausted.
            if (
                api_status == "success"
                and not api_client.quota_exhausted
                and video_id in unavailable_by_api_set
                and video_id in ids_missing_from_json_set
                and REMOVE_CONFIRMED_UNAVAILABLE_IDS_FROM_SEEN_TXT
            ):
                if video_id not in confirmed_unavailable_seen:
                    confirmed_unavailable_seen.add(video_id)
                    confirmed_unavailable_to_remove.append(video_id)
                    api_confirmed_unavailable_count += 1
                continue

            metadata, pytube_status = pytube.fetch(video_id)
            if metadata is not None:
                if video_id not in data:
                    data[video_id] = _empty_metadata(status="created_before_pytube_merge")
                _merge_metadata(
                    data[video_id],
                    metadata,
                    overwrite_views_likes=REFRESH_EXISTING_VIEWS_LIKES,
                )
                pytube_recovered_count += 1
                continue

            # pytubefix unavailable is used for text cleanup only when API quota
            # has not been exhausted. HTTP 429/rate limits never remove IDs.
            if (
                pytube_status == "unavailable"
                and not api_client.quota_exhausted
                and video_id in ids_missing_from_json_set
                and REMOVE_CONFIRMED_UNAVAILABLE_IDS_FROM_SEEN_TXT
            ):
                if video_id not in confirmed_unavailable_seen:
                    confirmed_unavailable_seen.add(video_id)
                    confirmed_unavailable_to_remove.append(video_id)
                    pytube_confirmed_unavailable_count += 1
                continue

            if video_id not in data and ADD_PLACEHOLDER_FOR_UNAVAILABLE_IDS:
                data[video_id] = _empty_metadata(status="not_recovered")
                placeholder_count += 1

        if CHECKPOINT_EVERY_N_BATCHES > 0 and batch_index % CHECKPOINT_EVERY_N_BATCHES == 0:
            _save_checkpoint(output_json_path, data)

        if SLEEP_SECONDS_BETWEEN_BATCHES > 0:
            time.sleep(SLEEP_SECONDS_BETWEEN_BATCHES)

    _apply_channel_average_views(data, api_client)

    if ADD_PLACEHOLDER_FOR_UNAVAILABLE_IDS:
        for video_id in seen_ids:
            if video_id not in data and video_id not in confirmed_unavailable_seen:
                data[video_id] = _empty_metadata(status="not_recovered_after_full_run")
                placeholder_count += 1

    removed_from_seen_files = 0
    if api_client.quota_exhausted:
        logger.warning(
            "YouTube Data API quota was exhausted during this run. "
            "Skipping all removals from seen text files for safety."
        )
    elif REMOVE_CONFIRMED_UNAVAILABLE_IDS_FROM_SEEN_TXT and confirmed_unavailable_to_remove:
        removed_from_seen_files = _remove_ids_from_seen_files(folder, confirmed_unavailable_to_remove)

    _atomic_save_json(output_json_path, data)

    logger.info("Recovery summary:")
    logger.info("  Seen IDs loaded from text files: {}".format(len(seen_ids)))
    logger.info("  Original JSON entries: {}".format(original_json_count))
    logger.info("  Final JSON entries: {}".format(len(data)))
    logger.info("  Recovered via API: {}".format(api_recovered_count))
    logger.info("  Recovered via pytubefix: {}".format(pytube_recovered_count))
    logger.info("  Confirmed unavailable by API: {}".format(api_confirmed_unavailable_count))
    logger.info("  Confirmed unavailable by pytubefix: {}".format(pytube_confirmed_unavailable_count))
    logger.info("  Placeholder entries added: {}".format(placeholder_count))
    logger.info("  Confirmed unavailable ID occurrences removed from text files: {}".format(removed_from_seen_files))
    logger.info("  Output JSON: {}".format(output_json_path))


if __name__ == "__main__":
    recover_json_from_seen_ids()
