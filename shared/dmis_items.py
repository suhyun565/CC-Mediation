"""
DMIS stage Likert items, derived from Bennett's official IDRI document:

  Bennett, M. J. (1986, 1993, 2002, 2005, 2011). A Developmental Model of
  Intercultural Sensitivity. Intercultural Development Research Institute.
  https://www.idrinstitute.org/

Each stage is operationalized as a set of first-person statements that
Bennett himself published as the "At this stage, learners say:" diagnostic
quotes. Some quotes are reworded slightly to remove travel-specific contexts
("ordering in restaurants", "study abroad", etc.) so that the items
generalize across the cultural-conflict scenarios in our dialogues.

Response scale: 7-point Likert
  1 = strongly disagree
  2 = disagree
  3 = slightly disagree
  4 = neutral
  5 = slightly agree
  6 = agree
  7 = strongly agree

Stage scoring:
  raw stage score   = mean of the stage's items (range [1, 7])
  ethnocentric sum  = D + Df + M
  ethnorelative sum = A + Ad + I
  composite IDI-style score (Paige et al. 2003 weighting):
      Σ w_k * v_k / (Σ v_k - 6)
      with weights w = (-3, -2, -1, +1, +2, +3) for (D, Df, M, A, Ad, I)
"""

from __future__ import annotations

DMIS_ITEMS: dict[str, list[str]] = {
    "Denial": [
        "Live and let live, that's what I say.",
        "As long as we all speak the same language, there's no problem.",
        "All I really need to know is what we are talking about — I can figure out the rest as I go along.",
        "With my experience, I can be successful in any cultural situation without any special effort.",
        "Different cultures are mostly the same once you get past the surface.",
        "I do not really see meaningful cultural differences between this person and me.",
    ],
    "Defense": [
        "Why don't these people just communicate the way I do?",
        "When I think about other cultures, I realize how much better my own culture is.",
        "My culture's way of handling this kind of situation should be a model for others.",
        "These people don't value things the way we do.",
        "We could really teach these people a lot.",
        "People from my cultural background tend to be more reasonable about this kind of issue.",
    ],
    "Minimization": [
        "The key to getting along in any cultural situation is to just be yourself — authentic and honest.",
        "Customs differ, of course, but when you really get to know people they're pretty much like us.",
        "I have an intuitive sense of other people, no matter what their cultural background.",
        "While the context may be different, the basic need to communicate remains the same everywhere.",
        "No matter what their culture, people are pretty much motivated by the same things.",
        "If people are really honest, they'll recognize that some values are universal.",
        "Deep down, we are all the same.",
        "It's a small world, after all.",
    ],
    "Acceptance": [
        "The more cultural difference there is here, the better — it would be boring if everyone saw this the same way.",
        "This person sees this situation in ways I had not thought of before.",
        "I try to understand the other person's cultural background before drawing conclusions about what they meant.",
        "The more I learn about how this person sees things culturally, the better I can understand our disagreement.",
        "Sometimes it's confusing, knowing that values differ across cultures and wanting to be respectful, but still wanting to maintain my own core values.",
        "It is appropriate that this person does not necessarily share the same values and goals as people from my own culture.",
        "Their cultural framing of this situation is a coherent way of seeing things, even though it differs from mine.",
        "I am genuinely curious about what cultural assumptions are shaping the other person's view here.",
    ],
    "Adaptation": [
        "To resolve this disagreement, I am going to have to change my approach.",
        "I know they are really trying to adapt to my style, so it is fair that I try to meet them halfway.",
        "I would interact with this person somewhat differently than I would with people from my own culture, to account for differences in how respect is communicated.",
        "I can maintain my own values and also behave in ways that are appropriate to their culture.",
        "To resolve this dispute, I need to change my behavior to account for the cultural difference between me and my counterpart.",
        "I find myself shifting my perspective into how this person sees the situation when I am with them.",
        "I can adjust my communication style to better connect with someone from a different cultural background.",
        "When I come into contact with someone from a different culture, I find I change my behavior to better engage with theirs.",
    ],
    "Integration": [
        "I am able to move between different cultural perspectives with relative ease.",
        "I feel most comfortable when I am bridging differences between the cultures I know.",
        "Whatever the situation, I can usually look at it from a variety of cultural points of view.",
        "My decision-making is enhanced by having multiple cultural frames of reference.",
        "I can fully participate in more than one cultural community.",
        "I see myself as someone whose identity draws from more than one culture.",
        "In an intercultural situation, I prefer drawing on multiple cultural perspectives rather than committing to just one.",
    ],
}

# Order matters for the composite score weighting.
STAGE_ORDER: list[str] = [
    "Denial",
    "Defense",
    "Minimization",
    "Acceptance",
    "Adaptation",
    "Integration",
]

# Paige et al. (2003) weights for the composite IDI-style developmental score.
COMPOSITE_WEIGHTS: dict[str, int] = {
    "Denial": -3,
    "Defense": -2,
    "Minimization": -1,
    "Acceptance": +1,
    "Adaptation": +2,
    "Integration": +3,
}

LIKERT_RANGE: tuple[int, int] = (1, 7)


def flat_items() -> list[tuple[str, int, str]]:
    """Return [(stage, item_index, text), ...] in stable order."""
    out: list[tuple[str, int, str]] = []
    for stage in STAGE_ORDER:
        for i, text in enumerate(DMIS_ITEMS[stage], start=1):
            out.append((stage, i, text))
    return out


def total_n_items() -> int:
    return sum(len(DMIS_ITEMS[s]) for s in STAGE_ORDER)


if __name__ == "__main__":
    print(f"Total DMIS Likert items: {total_n_items()}")
    for stage in STAGE_ORDER:
        print(f"  {stage:<14} {len(DMIS_ITEMS[stage])} items")