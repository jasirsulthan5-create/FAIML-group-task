"""Module 5 - Content AI evaluation: readability (Flesch Reading Ease),
weighted scorecard (Table 5.3), rubric scoring (Table 5.2), Cohen's kappa."""
import re
from collections import Counter

WEIGHTS = {  # Table 5.3 (suggested values)
    "web_article": {"clarity": 0.55, "tone": 0.45, "delivery": 0.0},
    "spoken_narration": {"clarity": 0.40, "tone": 0.35, "delivery": 0.25},
    "avatar_video": {"clarity": 0.30, "tone": 0.25, "delivery": 0.45},
}
DELIVERY_CUES = ["facial_expression", "gaze", "gestures",
                 "posture", "vocal_delivery", "lip_sync"]


def count_syllables(word):
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups)
    if word.endswith("e") and not word.endswith("le") and n > 1:
        n -= 1
    return max(n, 1)


def flesch_reading_ease(text):
    sentences = max(len(re.findall(r"[.!?]+", text)), 1)
    words = re.findall(r"[A-Za-z']+", text)
    syllables = sum(count_syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / sentences) \
        - 84.6 * (syllables / len(words))


def delivery_score(cue_scores):
    """Mean of 1-5 cue scores."""
    return sum(cue_scores[c] for c in DELIVERY_CUES) / len(DELIVERY_CUES)


def weighted_scorecard(fmt, clarity, tone, delivery=0.0):
    w = WEIGHTS[fmt]
    return (w["clarity"] * clarity + w["tone"] * tone
            + w["delivery"] * delivery)


def cohens_kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / n ** 2
    return (po - pe) / (1 - pe)


if __name__ == "__main__":
    sample = ("Welcome to our new AI laboratory. Here you will build real "
              "projects, work with friendly mentors, and learn by doing. "
              "Apply today and start your journey with us.")
    print(f"Flesch Reading Ease: {flesch_reading_ease(sample):.1f}")

    scores = dict(facial_expression=4, gaze=3, gestures=3,
                  posture=4, vocal_delivery=4, lip_sync=3)
    d = delivery_score(scores)
    print(f"Delivery score (1-5): {d:.2f}")
    print("Avatar video scorecard:",
          round(weighted_scorecard("avatar_video", 4.2, 4.0, d), 2))

    r1 = [5, 4, 3, 4, 5, 3, 4, 2, 4, 5]   # evaluator 1
    r2 = [5, 4, 3, 3, 5, 3, 4, 2, 4, 4]   # evaluator 2
    print(f"Cohen's kappa: {cohens_kappa(r1, r2):.2f}")
