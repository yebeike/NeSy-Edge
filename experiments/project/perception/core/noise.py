"""Project-owned semantic noise policy for perception evaluation."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


NOISE_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _stable_rng(
    seed: int,
    dataset_id: str,
    text: str,
    noise_level: float,
) -> random.Random:
    material = f"{seed}|{dataset_id}|{noise_level:.1f}|{text}".encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()[:16]
    return random.Random(int(digest, 16))


@dataclass(frozen=True)
class SemanticNoisePolicy:
    """Deterministic semantic noise injector used by project-owned evaluation."""

    seed: int = 2026

    def inject(
        self,
        clean_alert: str,
        dataset_id: str,
        noise_level: float,
    ) -> str:
        dataset_key = dataset_id.lower()
        if noise_level <= 0.0:
            return clean_alert
        if dataset_key == "hdfs":
            return _inject_hdfs(clean_alert, noise_level, self.seed)
        if dataset_key == "openstack":
            return _inject_openstack(clean_alert, noise_level, self.seed)
        if dataset_key == "hadoop":
            return _inject_hadoop(clean_alert, noise_level, self.seed)
        return clean_alert


def _replace_many(text: str, replacements: list[tuple[str, str]]) -> str:
    value = text
    for source, target in replacements:
        if source in value:
            value = value.replace(source, target)
    return value


def _inject_hdfs(text: str, noise_level: float, seed: int) -> str:
    value = text
    rng = _stable_rng(seed, "hdfs", text, noise_level)
    if rng.random() > noise_level:
        return value

    base_replacements = [
        ("PacketResponder", "PkgResponder"),
        ("terminating", "closing"),
        ("Received block", "Got blk"),
        ("Exception", "Error"),
        ("size", "len"),
        ("Receiving block", "replica segment ingress"),
        ("Verification succeeded", "integrity check passed"),
        ("Served block", "delivered replica segment"),
    ]
    value = _replace_many(value, base_replacements)
    if "blk_" in value:
        value = value.replace("blk_", "block-id:")

    if noise_level >= 0.6:
        value = _replace_many(
            value,
            [
                ("NameSystem.allocateBlock", "NameSystem.reserveTargets"),
                ("allocateBlock", "reserveTargets"),
                ("NameSystem.addStoredBlock", "NameSystem.registerReplica"),
                ("blockMap updated", "replica map refreshed"),
                ("is added to", "is mapped onto"),
                ("src:", "source="),
                ("dest:", "target="),
            ],
        )
    if noise_level >= 0.8:
        value = _replace_many(
            value,
            [
                ("replica segment ingress", "replica fragment ingress"),
                ("delivered replica segment", "served replica fragment"),
                ("integrity check passed", "replica verification passed"),
            ],
        )
    if noise_level >= 1.0:
        value = _replace_many(
            value,
            [
                ("NameSystem.reserveTargets", "NameSystem.reserveReplicaTargets"),
                ("reserveTargets", "reserveReplicaTargets"),
                ("NameSystem.registerReplica", "NameSystem.commitReplica"),
                ("replica map refreshed", "replica ledger refreshed"),
                ("PkgResponder", "AckStage"),
                ("Got blk", "replica fragment committed"),
                ("closing", "finalizing stream"),
            ],
        )
    return value


def _inject_openstack(text: str, noise_level: float, seed: int) -> str:
    value = text
    rng = _stable_rng(seed, "openstack", text, noise_level)
    apply_prob = min(1.0, 0.25 + 0.65 * noise_level)
    if rng.random() > apply_prob:
        return value

    rule_packs = [
        [('"GET ', '"FETCH '), ('"POST ', '"SUBMIT ')],
        [
            ("Claim successful", "Reservation accepted"),
            ("Creating image", "Generating snapshot"),
            ("Terminating instance", "Shutting down guest"),
            ("Instance destroyed successfully.", "Guest removal completed."),
            ("Deletion of ", "Removal of "),
            ("Took ", "Required "),
            ("Total disk: ", "Disk footprint: "),
        ],
        [
            ("[instance:", "[guest:"),
            (" instance", " guest"),
            ("instances", "guests"),
            ("While synchronizing instance power states", "While reconciling guest power states"),
        ],
        [
            (" status: ", " state="),
            (" len: ", " bytes="),
            (" time: ", " duration="),
        ],
        [
            ("HTTP exception thrown:", "API fault raised:"),
            ("No instances found for any event", "No guests matched any event"),
            ("VM Started", "Guest booted"),
            ("VM Stopped", "Guest halted"),
            ("VM Paused", "Guest paused"),
            ("VM Resumed", "Guest resumed"),
            ("Unknown base file", "Unresolved base image"),
            ("network-vif-plugged", "vif-attached"),
            ("network-vif-unplugged", "vif-detached"),
        ],
    ]
    packs_to_apply = min(len(rule_packs), max(1, int(round(noise_level * 5.0))))
    applied = 0
    for pack in rule_packs:
        if applied >= packs_to_apply:
            break
        if not any(source in value for source, _ in pack):
            continue
        value = _replace_many(value, pack)
        applied += 1

    if noise_level >= 0.6:
        value = _replace_many(
            value,
            [
                ("Lifecycle Event", "Lifecycle notice"),
                ("successful", "accepted"),
            ],
        )
    if noise_level >= 0.8:
        value = _replace_many(
            value,
            [
                ("INFO", "NOTE"),
                ("warning", "notice"),
            ],
        )
    if noise_level >= 1.0:
        value = value.replace("error", "fault")
    return value


def _inject_hadoop(text: str, noise_level: float, seed: int) -> str:
    value = text
    rng = _stable_rng(seed, "hadoop", text, noise_level)
    apply_prob = min(1.0, 0.25 + 0.65 * noise_level)
    if rng.random() > apply_prob:
        return value

    rule_packs = [
        [
            ("Failed to renew lease for", "Lease refresh failed for"),
            ("ERROR IN CONTACTING RM.", "RM contact fault."),
            ("Address change detected.", "Endpoint remap observed."),
            ("Recalculating schedule,", "Recomputing plan,"),
            ("Reduce slow start threshold not met.", "Reduce ramp threshold pending."),
        ],
        [
            ("Launching attempt_", "Bootstrapping attempt_"),
            ("ATTEMPT_START", "ATT_START"),
            ("JVM with ID:", "Worker JVM:"),
            ("given task:", "assigned work:"),
            ("Task Transitioned from NEW to SCHEDULED", "Task state moved NEW->SCHEDULED"),
        ],
        [
            ("attempt_", "att_"),
            ("TaskAttempt", "TkAttempt"),
            ("Task", "Tk"),
            ("task_", "tk_"),
            ("application_", "app_"),
            ("Application", "App"),
        ],
    ]
    packs_to_apply = min(len(rule_packs), max(1, int(round(noise_level * len(rule_packs)))))
    applied = 0
    for pack in rule_packs:
        if applied >= packs_to_apply:
            break
        if not any(source in value for source, _ in pack):
            continue
        value = _replace_many(value, pack)
        applied += 1
    return value
