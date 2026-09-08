"""Generation III (Pokémon FireRed / BPRD) type system.

This module is **pure data + pure functions**. It has no RAM, no emulator, no
PPO dependency and is safe to import anywhere.

Source of the constants below: the Generation III games are unambiguous and
identical across languages for battle mechanics. The type IDs match
``pret/pokefirered`` ``include/constants/pokemon.h`` (``TYPE_*``). The
type-effectiveness chart is the canonical Gen-I..V chart *before* the Gen-VI
Fairy type and *before* the Gen-II Steel/Ghost/Dark corrections were... (they
were already in by Gen II, so the Gen-III chart used here already contains
Steel resisting Dark/Ghost and Ghost/Dark hitting each other for neutral - see
the notes on each pair). Ghost vs Psychic is 2x (fixed since Gen II).

Physical / Special in Gen III is decided by the MOVE'S TYPE, not by a per-move
category flag (that split arrived in Gen IV). ``PHYSICAL_TYPES`` /
``SPECIAL_TYPES`` below encode the Gen-III rule.
"""
from __future__ import annotations

# -- type ids (Gen III internal) ------------------------------------------
TYPE_NORMAL = 0
TYPE_FIGHTING = 1
TYPE_FLYING = 2
TYPE_POISON = 3
TYPE_GROUND = 4
TYPE_ROCK = 5
TYPE_BUG = 6
TYPE_GHOST = 7
TYPE_STEEL = 8
TYPE_MYSTERY = 9   # "???" - the Curse type; only ever appears on struggle/curse
TYPE_FIRE = 10
TYPE_WATER = 11
TYPE_GRASS = 12
TYPE_ELECTRIC = 13
TYPE_PSYCHIC = 14
TYPE_ICE = 15
TYPE_DRAGON = 16
TYPE_DARK = 17

TYPE_COUNT = 18   # 0..17; index 9 (MYSTERY) is reserved but kept for shape

TYPE_NAMES = {
    TYPE_NORMAL: "Normal", TYPE_FIGHTING: "Fighting", TYPE_FLYING: "Flying",
    TYPE_POISON: "Poison", TYPE_GROUND: "Ground", TYPE_ROCK: "Rock",
    TYPE_BUG: "Bug", TYPE_GHOST: "Ghost", TYPE_STEEL: "Steel",
    TYPE_MYSTERY: "???", TYPE_FIRE: "Fire", TYPE_WATER: "Water",
    TYPE_GRASS: "Grass", TYPE_ELECTRIC: "Electric", TYPE_PSYCHIC: "Psychic",
    TYPE_ICE: "Ice", TYPE_DRAGON: "Dragon", TYPE_DARK: "Dark",
}

# Gen III physical/special split is BY TYPE.
PHYSICAL_TYPES = frozenset({
    TYPE_NORMAL, TYPE_FIGHTING, TYPE_FLYING, TYPE_POISON, TYPE_GROUND,
    TYPE_ROCK, TYPE_BUG, TYPE_GHOST, TYPE_STEEL, TYPE_MYSTERY,
})
SPECIAL_TYPES = frozenset({
    TYPE_FIRE, TYPE_WATER, TYPE_GRASS, TYPE_ELECTRIC, TYPE_PSYCHIC,
    TYPE_ICE, TYPE_DRAGON, TYPE_DARK,
})

# -- effectiveness chart -------------------------------------------------
# Only non-1x matchups are listed. (attacker, defender) -> multiplier.
# Anything not present is 1.0 (neutral). Values are the Gen-III canon:
#   0.0  = no effect / immune
#   0.5  = not very effective
#   2.0  = super effective
_CHART = {
    # NORMAL
    (TYPE_NORMAL, TYPE_ROCK): 0.5, (TYPE_NORMAL, TYPE_STEEL): 0.5,
    (TYPE_NORMAL, TYPE_GHOST): 0.0,
    # FIGHTING
    (TYPE_FIGHTING, TYPE_NORMAL): 2.0, (TYPE_FIGHTING, TYPE_ROCK): 2.0,
    (TYPE_FIGHTING, TYPE_STEEL): 2.0, (TYPE_FIGHTING, TYPE_ICE): 2.0,
    (TYPE_FIGHTING, TYPE_DARK): 2.0,
    (TYPE_FIGHTING, TYPE_FLYING): 0.5, (TYPE_FIGHTING, TYPE_POISON): 0.5,
    (TYPE_FIGHTING, TYPE_BUG): 0.5, (TYPE_FIGHTING, TYPE_PSYCHIC): 0.5,
    (TYPE_FIGHTING, TYPE_GHOST): 0.0,
    # FLYING
    (TYPE_FLYING, TYPE_FIGHTING): 2.0, (TYPE_FLYING, TYPE_BUG): 2.0,
    (TYPE_FLYING, TYPE_GRASS): 2.0,
    (TYPE_FLYING, TYPE_ROCK): 0.5, (TYPE_FLYING, TYPE_STEEL): 0.5,
    (TYPE_FLYING, TYPE_ELECTRIC): 0.5,
    # POISON
    (TYPE_POISON, TYPE_GRASS): 2.0,
    (TYPE_POISON, TYPE_POISON): 0.5, (TYPE_POISON, TYPE_GROUND): 0.5,
    (TYPE_POISON, TYPE_ROCK): 0.5, (TYPE_POISON, TYPE_GHOST): 0.5,
    (TYPE_POISON, TYPE_STEEL): 0.0,
    # GROUND
    (TYPE_GROUND, TYPE_POISON): 2.0, (TYPE_GROUND, TYPE_ROCK): 2.0,
    (TYPE_GROUND, TYPE_STEEL): 2.0, (TYPE_GROUND, TYPE_FIRE): 2.0,
    (TYPE_GROUND, TYPE_ELECTRIC): 2.0,
    (TYPE_GROUND, TYPE_BUG): 0.5, (TYPE_GROUND, TYPE_GRASS): 0.5,
    (TYPE_GROUND, TYPE_FLYING): 0.0,
    # ROCK
    (TYPE_ROCK, TYPE_FLYING): 2.0, (TYPE_ROCK, TYPE_BUG): 2.0,
    (TYPE_ROCK, TYPE_FIRE): 2.0, (TYPE_ROCK, TYPE_ICE): 2.0,
    (TYPE_ROCK, TYPE_FIGHTING): 0.5, (TYPE_ROCK, TYPE_GROUND): 0.5,
    (TYPE_ROCK, TYPE_STEEL): 0.5,
    # BUG
    (TYPE_BUG, TYPE_GRASS): 2.0, (TYPE_BUG, TYPE_PSYCHIC): 2.0,
    (TYPE_BUG, TYPE_DARK): 2.0,
    (TYPE_BUG, TYPE_FIGHTING): 0.5, (TYPE_BUG, TYPE_FLYING): 0.5,
    (TYPE_BUG, TYPE_POISON): 0.5, (TYPE_BUG, TYPE_GHOST): 0.5,
    (TYPE_BUG, TYPE_STEEL): 0.5, (TYPE_BUG, TYPE_FIRE): 0.5,
    # GHOST
    (TYPE_GHOST, TYPE_GHOST): 2.0, (TYPE_GHOST, TYPE_PSYCHIC): 2.0,
    (TYPE_GHOST, TYPE_DARK): 0.5, (TYPE_GHOST, TYPE_STEEL): 0.5,
    (TYPE_GHOST, TYPE_NORMAL): 0.0,
    # STEEL
    (TYPE_STEEL, TYPE_ROCK): 2.0, (TYPE_STEEL, TYPE_ICE): 2.0,
    (TYPE_STEEL, TYPE_STEEL): 0.5, (TYPE_STEEL, TYPE_FIRE): 0.5,
    (TYPE_STEEL, TYPE_WATER): 0.5, (TYPE_STEEL, TYPE_ELECTRIC): 0.5,
    # FIRE
    (TYPE_FIRE, TYPE_BUG): 2.0, (TYPE_FIRE, TYPE_STEEL): 2.0,
    (TYPE_FIRE, TYPE_GRASS): 2.0, (TYPE_FIRE, TYPE_ICE): 2.0,
    (TYPE_FIRE, TYPE_ROCK): 0.5, (TYPE_FIRE, TYPE_FIRE): 0.5,
    (TYPE_FIRE, TYPE_WATER): 0.5, (TYPE_FIRE, TYPE_DRAGON): 0.5,
    # WATER
    (TYPE_WATER, TYPE_GROUND): 2.0, (TYPE_WATER, TYPE_ROCK): 2.0,
    (TYPE_WATER, TYPE_FIRE): 2.0,
    (TYPE_WATER, TYPE_WATER): 0.5, (TYPE_WATER, TYPE_GRASS): 0.5,
    (TYPE_WATER, TYPE_DRAGON): 0.5,
    # GRASS
    (TYPE_GRASS, TYPE_GROUND): 2.0, (TYPE_GRASS, TYPE_ROCK): 2.0,
    (TYPE_GRASS, TYPE_WATER): 2.0,
    (TYPE_GRASS, TYPE_FLYING): 0.5, (TYPE_GRASS, TYPE_POISON): 0.5,
    (TYPE_GRASS, TYPE_BUG): 0.5, (TYPE_GRASS, TYPE_STEEL): 0.5,
    (TYPE_GRASS, TYPE_FIRE): 0.5, (TYPE_GRASS, TYPE_GRASS): 0.5,
    (TYPE_GRASS, TYPE_DRAGON): 0.5,
    # ELECTRIC
    (TYPE_ELECTRIC, TYPE_FLYING): 2.0, (TYPE_ELECTRIC, TYPE_WATER): 2.0,
    (TYPE_ELECTRIC, TYPE_GRASS): 0.5, (TYPE_ELECTRIC, TYPE_ELECTRIC): 0.5,
    (TYPE_ELECTRIC, TYPE_DRAGON): 0.5,
    (TYPE_ELECTRIC, TYPE_GROUND): 0.0,
    # PSYCHIC
    (TYPE_PSYCHIC, TYPE_FIGHTING): 2.0, (TYPE_PSYCHIC, TYPE_POISON): 2.0,
    (TYPE_PSYCHIC, TYPE_PSYCHIC): 0.5, (TYPE_PSYCHIC, TYPE_STEEL): 0.5,
    (TYPE_PSYCHIC, TYPE_DARK): 0.0,
    # ICE
    (TYPE_ICE, TYPE_FLYING): 2.0, (TYPE_ICE, TYPE_GROUND): 2.0,
    (TYPE_ICE, TYPE_GRASS): 2.0, (TYPE_ICE, TYPE_DRAGON): 2.0,
    (TYPE_ICE, TYPE_STEEL): 0.5, (TYPE_ICE, TYPE_FIRE): 0.5,
    (TYPE_ICE, TYPE_WATER): 0.5, (TYPE_ICE, TYPE_ICE): 0.5,
    # DRAGON
    (TYPE_DRAGON, TYPE_DRAGON): 2.0,
    (TYPE_DRAGON, TYPE_STEEL): 0.5,
    # DARK
    (TYPE_DARK, TYPE_GHOST): 2.0, (TYPE_DARK, TYPE_PSYCHIC): 2.0,
    (TYPE_DARK, TYPE_FIGHTING): 0.5, (TYPE_DARK, TYPE_DARK): 0.5,
    (TYPE_DARK, TYPE_STEEL): 0.5,
}


def is_valid_type(t):
    return isinstance(t, int) and 0 <= t < TYPE_COUNT


def is_physical_type(move_type):
    """Gen III: physical/special is decided by the move's type."""
    return move_type in PHYSICAL_TYPES


def is_special_type(move_type):
    return move_type in SPECIAL_TYPES


def type_pair_multiplier(attack_type, defender_type):
    """One attacking type vs one defending type. 1.0 if not in the chart."""
    if not (is_valid_type(attack_type) and is_valid_type(defender_type)):
        return 1.0
    return _CHART.get((attack_type, defender_type), 1.0)


def effectiveness(attack_type, defender_types):
    """Product over the defender's (1 or 2) types.

    ``defender_types`` may be an int, or an iterable of ints. A duplicated
    second type (mono-type Pokémon store type2 == type1 in Gen III) is only
    counted once.
    Returns one of 0.0, 0.25, 0.5, 1.0, 2.0, 4.0.
    """
    if isinstance(defender_types, int):
        defender_types = (defender_types,)
    seen = []
    for t in defender_types:
        if is_valid_type(t) and t not in seen:
            seen.append(t)
    if not seen:
        return 1.0
    mult = 1.0
    for dt in seen:
        mult *= type_pair_multiplier(attack_type, dt)
    return mult


def stab_multiplier(move_type, attacker_types):
    """1.5 when the move's type matches one of the attacker's types, else 1.0."""
    if isinstance(attacker_types, int):
        attacker_types = (attacker_types,)
    if not is_valid_type(move_type):
        return 1.0
    return 1.5 if any(move_type == t for t in attacker_types
                      if is_valid_type(t)) else 1.0
