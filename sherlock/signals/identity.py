from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from typing import List, Optional

from ..models import (
    EvidencePacket,
    MeetingContext,
    Participant,
    SignalAxis,
    SignalSource,
)


def _normalize_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", "", name.lower().strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _fuzzy_name_match(name_a: str, name_b: str) -> float:
    a_tokens = set(_normalize_name(name_a).split())
    b_tokens = set(_normalize_name(name_b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    intersection = a_tokens & b_tokens
    union = a_tokens | b_tokens
    return len(intersection) / len(union) if union else 0.0


def _is_device_name(name: str) -> bool:
    device_patterns = [
        r"^(iphone|ipad|macbook|pixel|galaxy|oneplus|huawei|samsung|lg|sony)",
        r"(pro|max|plus|ultra|mini)$",
        r"^(android|ios|windows|linux|chrome)",
    ]
    lower = name.lower()
    return any(re.search(p, lower) for p in device_patterns)


class IdentitySignalExtractor:
    def __init__(self, context: MeetingContext):
        self.context = context

    def extract_calendar_match(self) -> List[EvidencePacket]:
        evidence = []
        now = datetime.utcnow()

        if not self.context.candidate_name and not self.context.candidate_email:
            return evidence

        for pid, participant in self.context.participants.items():
            score = 0.0
            rationales = []

            if self.context.candidate_name and participant.display_name:
                match_score = _fuzzy_name_match(
                    self.context.candidate_name, participant.display_name
                )
                if match_score > 0.5:
                    score += match_score * 2.0
                    rationales.append(
                        f"Name similarity {match_score:.2f} between "
                        f"'{self.context.candidate_name}' and '{participant.display_name}'"
                    )
                elif _is_device_name(participant.display_name):
                    rationales.append(
                        f"Display name '{participant.display_name}' appears to be a device name, "
                        f"not a person name"
                    )

            if self.context.candidate_email and participant.email:
                if self.context.candidate_email.lower() == participant.email.lower():
                    score += 3.0
                    rationales.append("Exact email match with calendar invite")
                elif participant.email:
                    email_user_a = self.context.candidate_email.split("@")[0]
                    email_user_b = participant.email.split("@")[0]
                    email_match = _fuzzy_name_match(email_user_a, email_user_b)
                    if email_match > 0.5:
                        score += email_match * 2.0
                        rationales.append(f"Email username similarity: {email_match:.2f}")

            if score > 0:
                evidence.append(EvidencePacket(
                    source=SignalSource.CALENDAR_MATCH,
                    axis=SignalAxis.IDENTITY,
                    target_participant_id=pid,
                    delta_log_odds=score,
                    confidence=min(score / 3.0, 1.0),
                    rationale="; ".join(rationales),
                    timestamp=now,
                    metadata={
                        "candidate_name": self.context.candidate_name,
                        "participant_name": participant.display_name,
                    },
                ))

        return evidence

    def extract_interviewer_negative(self) -> List[EvidencePacket]:
        evidence = []
        now = datetime.utcnow()

        interviewer_identifiers = set()
        for name in self.context.interviewer_names:
            interviewer_identifiers.add(_normalize_name(name))
        for email in self.context.interviewer_emails:
            interviewer_identifiers.add(email.lower())

        for pid, participant in self.context.participants.items():
            is_interviewer = False
            rationales = []

            participant_normalized = _normalize_name(participant.display_name)
            for interviewer_id in interviewer_identifiers:
                if interviewer_id in participant_normalized or participant_normalized in interviewer_id:
                    is_interviewer = True
                    rationales.append(
                        f"Name matches known interviewer: '{participant.display_name}'"
                    )
                    break

                if participant.email and interviewer_id == participant.email.lower():
                    is_interviewer = True
                    rationales.append(
                        f"Email matches known interviewer: '{participant.email}'"
                    )
                    break

            if is_interviewer:
                evidence.append(EvidencePacket(
                    source=SignalSource.INTERVIEWER_NEGATIVE,
                    axis=SignalAxis.IDENTITY,
                    target_participant_id=pid,
                    delta_log_odds=-2.5,
                    confidence=0.9,
                    rationale="; ".join(rationales),
                    timestamp=now,
                ))

        return evidence

    def extract_email_domain(self) -> List[EvidencePacket]:
        evidence = []
        now = datetime.utcnow()

        if not self.context.candidate_email:
            return evidence

        candidate_domain = self.context.candidate_email.split("@")[-1].lower()

        for pid, participant in self.context.participants.items():
            if not participant.email:
                continue

            participant_domain = participant.email.split("@")[-1].lower()

            interviewer_domains = {
                e.split("@")[-1].lower() for e in self.context.interviewer_emails
            }

            if participant_domain != candidate_domain and participant_domain not in interviewer_domains:
                evidence.append(EvidencePacket(
                    source=SignalSource.EMAIL_DOMAIN,
                    axis=SignalAxis.IDENTITY,
                    target_participant_id=pid,
                    delta_log_odds=0.2,
                    confidence=0.3,
                    rationale=(
                        f"External email domain '{participant_domain}' "
                        f"(candidate domain: '{candidate_domain}')"
                    ),
                    timestamp=now,
                ))

        return evidence

    def extract_display_name_change(self) -> List[EvidencePacket]:
        evidence = []
        now = datetime.utcnow()

        for pid, participant in self.context.participants.items():
            history = participant.display_name_history
            if not history or not self.context.candidate_name:
                continue

            calendar_normalized = _normalize_name(self.context.candidate_name)

            for change in history:
                old_name = change.get("old_name", "")
                if _fuzzy_name_match(old_name, calendar_normalized) > 0.5:
                    evidence.append(EvidencePacket(
                        source=SignalSource.DISPLAY_NAME_CHANGE,
                        axis=SignalAxis.IDENTITY,
                        target_participant_id=pid,
                        delta_log_odds=1.5,
                        confidence=0.7,
                        rationale=(
                            f"Display name changed from '{old_name}' (matches calendar name) "
                            f"to '{participant.display_name}'"
                        ),
                        timestamp=now,
                        metadata={"old_name": old_name, "new_name": participant.display_name},
                    ))
                    break

        return evidence

    def extract_join_timing(self) -> List[EvidencePacket]:
        evidence = []
        now = datetime.utcnow()

        if not self.context.scheduled_start:
            return evidence

        for pid, participant in self.context.participants.items():
            if not participant.join_time:
                continue

            join_offset = (participant.join_time - self.context.scheduled_start).total_seconds()

            if -300 <= join_offset <= 300:
                evidence.append(EvidencePacket(
                    source=SignalSource.JOIN_TIMING,
                    axis=SignalAxis.IDENTITY,
                    target_participant_id=pid,
                    delta_log_odds=1.0,
                    confidence=0.6,
                    rationale=(
                        f"Joined {join_offset:.0f}s relative to scheduled start "
                        f"(within expected window)"
                    ),
                    timestamp=now,
                    metadata={"join_offset_seconds": join_offset},
                ))
            elif join_offset > 1800:
                evidence.append(EvidencePacket(
                    source=SignalSource.JOIN_TIMING,
                    axis=SignalAxis.IDENTITY,
                    target_participant_id=pid,
                    delta_log_odds=-0.5,
                    confidence=0.3,
                    rationale=(
                        f"Joined {join_offset:.0f}s after scheduled start "
                        f"(significantly late)"
                    ),
                    timestamp=now,
                    metadata={"join_offset_seconds": join_offset},
                ))

        return evidence

    def extract_all(self) -> List[EvidencePacket]:
        evidence = []
        evidence.extend(self.extract_calendar_match())
        evidence.extend(self.extract_interviewer_negative())
        evidence.extend(self.extract_email_domain())
        evidence.extend(self.extract_join_timing())
        evidence.extend(self.extract_display_name_change())
        return evidence
