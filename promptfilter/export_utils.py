# export_utils.py
import json
from typing import Optional, Dict, Any, Set
from django.core.serializers.json import DjangoJSONEncoder
from django.db import transaction
from django.db.models import Prefetch

from .youtube import YoutubeAPI
from .models import (
    User, Channel, Video, Comment, PromptFilter, PromptRubric,
    Example, FilterPrediction
)
from typing import List, Dict, Any, Optional

def export_user_bundle(
    username: str,
    *,
    outfile: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Export a creator's entire bundle to JSON-friendly dict.

    Includes:
      - user
      - their channel (OneToOne)
      - videos under that channel
      - all comments under those videos (incl. other commenters)
      - filters on that channel (name and description only)
      - predictions (FilterPrediction) linked to those filters

    Notes on IDs:
      - username, channel.id, video.id, comment.id are exported as-is (you said these are globally unique).
      - PromptFilter has auto-increment IDs that may collide on the server; we export:
          * legacy_id: original PK (for your reference)
          * natural_key: { "name": <filter.name>, "channel_id": <channel.id> }
        Your import should match/create by the natural_key and ignore legacy_id.
    """
    with transaction.atomic():
        user = User.objects.get(username=username)

        # Build user payload (redact tokens by default)
        user_payload = {
            "username": user.username,
            "avatar": user.avatar,
            "whether_experiment": user.whether_experiment,
        }

        # Channel (may not exist yet)
        channel = user.channel
        channel_payload = None
        if channel:
            channel_payload = {
                "id": channel.id,
                "name": channel.name,
                "owner": user.username,
            }

        # Videos & Comments under this channel
        videos_payload = []
        comments_payload = []
        related_usernames: Set[str] = set()

        if channel:
            videos = (
                Video.objects.filter(channel=channel)
                .order_by("posted_at")
                .prefetch_related(
                    Prefetch(
                        "comments",
                        queryset=Comment.objects.select_related("user", "video")
                    )
                )
            )

            for v in videos:
                videos_payload.append({
                    "id": v.id,
                    "channel_id": channel.id,
                    "title": v.title,
                    "description": v.description,
                    "video_link": v.video_link,
                    "thumbnail": v.thumbnail,
                    "posted_at": v.posted_at,
                })

                for c in v.comments.all():
                    comments_payload.append({
                        "id": c.id,
                        "video_id": v.id,
                        "user": c.user.username,
                        "content": c.content,
                        "posted_at": c.posted_at
                    })
                    if c.user_id != user.username:
                        related_usernames.add(c.user.username)

        # Minimal profiles for *other* commenters we must create on import
        related_users_payload = []
        if related_usernames:
            other_users = User.objects.filter(username__in=related_usernames)
            for ou in other_users:
                related_users_payload.append({
                    "username": ou.username,
                    "avatar": ou.avatar,
                    "whether_experiment": ou.whether_experiment,
                })

        # Filters + rubrics + few-shot examples + predictions
        filters_payload = []
        predictions_payload = []

        if channel:
            filters = (
                PromptFilter.objects.filter(channel=channel)
                .prefetch_related(
                    Prefetch("matches", queryset=FilterPrediction.objects.select_related("comment", "comment__video"))
                )
            )

            for f in filters:
                filters_payload.append({
                    "legacy_id": f.id,  # may collide on server; do NOT rely on it when importing
                    "natural_key": {"name": f.name, "channel_id": f.channel_id},
                    "name": f.name,
                    "description": f.description,
                    "action": f.action,
                    "approach": f.approach,
                })

                # Predictions tied to this filter
                for m in f.matches.all():
                    predictions_payload.append({
                        "filter_natural_key": {"name": f.name, "channel_id": f.channel_id},
                        "comment_id": m.comment_id,
                        "prediction": m.prediction,
                        "confidence": m.confidence,
                        "groundtruth": m.groundtruth,
                        "experiment_type": m.experiment_type,
                    })

        bundle = {
            "user": user_payload,
            "channel": channel_payload,
            "videos": videos_payload,
            "comments": comments_payload,
            "related_users": related_users_payload,   # commenters other than the main user
            "filters": filters_payload,
            "predictions": predictions_payload,
        }

        if outfile:
            with open(outfile, "w") as f:
                json.dump(bundle, f, cls=DjangoJSONEncoder, indent=2)

        return bundle

# In export_utils.py (same file as export_user_bundle)
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

from django.db import transaction, IntegrityError

def _parse_dt(val):
    if not val:
        return None
    if isinstance(val, str):
        # Accept both "...Z" and "+00:00" forms
        s = val.rstrip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    return val

@transaction.atomic
def import_user_bundle(
    *,
    infile: Optional[str] = None,
    bundle: Optional[Dict[str, Any]] = None,
) -> Dict[str, int]:
    """
    Import a bundle produced by export_user_bundle().

    Idempotent:
      - Users, channel, videos, comments are update_or_create'd by their primary keys.
      - PromptFilters are matched/created by (name, channel_id) NATURAL KEY.
      - FilterPrediction is matched by (filter, comment) unique_together.

    Returns a summary dict of created/updated/skipped counts.
    """


    if (infile is None) == (bundle is None):
        raise ValueError("Provide exactly one of infile=... or bundle=...")

    if infile:
        with open(infile, "r") as f:
            data = json.load(f)
    else:
        data = bundle

    youtube = YoutubeAPI(credentials=None)
    # --- Caches & counters ---
    counts = dict(
        users_created=0, users_updated=0,
        channels_created=0, channels_updated=0,
        videos_created=0, videos_updated=0,
        comments_created=0, comments_updated=0, comments_parent_linked=0, comments_parent_missing=0,
        filters_created=0, filters_updated=0,
        predictions_created=0, predictions_updated=0, predictions_skipped_missing_comment=0,
    )
    user_cache: Dict[str, User] = {}


    # --- 1) Main user ---
    u = data["user"]
    user_obj, created = User.objects.update_or_create(
        username=u["username"],
        defaults={
            "avatar": u.get("avatar"),
            "whether_experiment": u.get("whether_experiment", False),
        },
    )
    user_cache[user_obj.username] = user_obj
    if created:
        counts["users_created"] += 1
    else:
        counts["users_updated"] += 1

    # --- 2) Channel (optional in bundle, but present in your export) ---
    channel_obj = None
    if data.get("channel"):
        ch = data["channel"]
        channel_obj, ch_created = Channel.objects.get_or_create(
            id=ch["id"],
            defaults={
                "owner": user_obj,
                "name": ch.get("name", ch["id"]),
            },
        )
        # Keep things in sync if it already existed
        updated = False
        if channel_obj.owner_id != user_obj.username:
            channel_obj.owner = user_obj
            updated = True
        nm = ch.get("name")
        if nm and nm != channel_obj.name:
            channel_obj.name = nm
            updated = True
        if updated:
            channel_obj.save()
            counts["channels_updated"] += 1
        elif ch_created:
            counts["channels_created"] += 1

    # --- 3) Related users (other commenters) ---
    for ru in data.get("related_users", []):
        ru_obj, ru_created = User.objects.update_or_create(
            username=ru["username"],
            defaults={
                "avatar": ru.get("avatar"),
                "whether_experiment": ru.get("whether_experiment", False),
            },
        )
        user_cache[ru_obj.username] = ru_obj
        if ru_created:
            counts["users_created"] += 1
        else:
            counts["users_updated"] += 1

    # Ensure main user in cache (if not already)
    user_cache.setdefault(user_obj.username, user_obj)

    # --- 4) Videos ---
    video_cache: Dict[str, Video] = {}
    for v in data.get("videos", []):
        posted_at = _parse_dt(v.get("posted_at"))
        vid_obj, vid_created = Video.objects.update_or_create(
            id=v["id"],
            defaults={
                "channel": channel_obj,
                "title": v.get("title", ""),
                "description": v.get("description"),
                "video_link": v.get("video_link", ""),
                "thumbnail": v.get("thumbnail"),
                "posted_at": posted_at,
            },
        )
        video_cache[vid_obj.id] = vid_obj
        if vid_created:
            counts["videos_created"] += 1
        else:
            counts["videos_updated"] += 1

    # --- 5) Comments (two-pass to handle parents) ---
    # Pass A: create/update without parent
    for c in data.get("comments", []):
        c_user = user_cache.get(c["user"])
        if not c_user:
            # User must exist; skip comment otherwise (should not happen)
            continue

        c_video = video_cache.get(c["video_id"])
        if not c_video:
            # Video must exist; skip comment otherwise (should not happen)
            continue

        posted_at = _parse_dt(c.get("posted_at"))
        com_obj, com_created = Comment.objects.update_or_create(
            id=c["id"],
            defaults={
                "video": c_video,
                "user": c_user,
                "content": c.get("content", ""),
                "posted_at": posted_at,
            },
        )
        if com_created:
            counts["comments_created"] += 1
        else:
            counts["comments_updated"] += 1

    # --- 6) Filters (by natural key: name + channel_id) ---
    filter_cache: Dict[Tuple[str, str], PromptFilter] = {}
    for f in data.get("filters", []):
        nk = f["natural_key"]
        key = (nk["name"], nk["channel_id"])
        # Ensure the channel exists (it should—it's same as channel_obj)
        if not Channel.objects.filter(id=nk["channel_id"]).exists():
            # If somehow missing, skip this filter
            continue

        filt_obj, filt_created = PromptFilter.objects.get_or_create(
            name=nk["name"],
            channel_id=nk["channel_id"],
            defaults={
                "description": f.get("description", ""),
                "approach": f.get("approach"),
            },
        )

        filter_cache[key] = filt_obj

    # --- 7) Predictions ---
    for p in data.get("predictions", []):
        nk = p["filter_natural_key"]
        key = (nk["name"], nk["channel_id"])
        filt_obj = filter_cache.get(key)
        if not filt_obj:
            # If filter missing (unexpected), skip
            continue

        try:
            com_obj = Comment.objects.get(id=p["comment_id"])
        except Comment.DoesNotExist:
            counts["predictions_skipped_missing_comment"] += 1
            continue

        pred_obj, pred_created = FilterPrediction.objects.update_or_create(
            filter=filt_obj,
            comment=com_obj,
            defaults={
                "prediction": p.get("prediction"),
                "confidence": p.get("confidence"),
                "groundtruth": p.get("groundtruth"),
                "experiment_type": p.get("experiment_type"),
            },
        )
        if pred_created:
            counts["predictions_created"] += 1
        else:
            counts["predictions_updated"] += 1

    return counts

def export_bundles_simple(usernames: List[str], outfile: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Build a list of per-user bundles using export_user_bundle() and (optionally) dump to JSON.
    Output format: {"bundles": [ <bundle>, <bundle>, ... ]}
    """
    bundles = [export_user_bundle(u) for u in usernames]
    if outfile:
        with open(outfile, "w") as f:
            json.dump({"bundles": bundles}, f, cls=DjangoJSONEncoder, indent=2)
    return bundles


def import_bundles_simple(infile: Optional[str] = None, bundles: Optional[List[Dict[str, Any]]] = None):
    """
    Iterate over the list of bundles and import each with import_user_bundle().
    Provide exactly one of infile or bundles.
    Returns the list of per-bundle count dicts from import_user_bundle().
    """
    if (infile is None) == (bundles is None):
        raise ValueError("Provide exactly one of infile=... or bundles=[...].")

    if infile:
        with open(infile, "r") as f:
            data = json.load(f)
        bundles = data.get("bundles", [])

    results = []
    for b in bundles:
        results.append(import_user_bundle(bundle=b))
    return results