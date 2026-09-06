"""Conservative battle tracking for FireRed's persistent enemy-party RAM."""


def enemy_fingerprint(party):
    return tuple(
        (
            int(mon.get("slot", -1)),
            int(mon.get("species_id", 0)),
            int(mon.get("personality", 0)),
            int(mon.get("cur_hp", 0)),
            tuple(
                (int(move.get("id", 0)), int(move.get("pp", 0)))
                for move in mon.get("moves", [])
            ),
        )
        for mon in party or ()
    )


class BattleState:
    def __init__(self, party=(), position=None):
        self.fingerprint = enemy_fingerprint(party)
        self.position = position
        self.active = False
        self.raw_flags = 0
        self.reason = "reset"
        self.previous_signal = False

    def update(self, party, position, flags=0, signal=False, live=None):
        fingerprint = enemy_fingerprint(party)
        fresh_enemy = bool(fingerprint and fingerprint != self.fingerprint)
        moved = (
            position is not None
            and self.position is not None
            and position != self.position
        )
        self.raw_flags = int(flags or 0)
        rising = bool(signal) and not self.previous_signal
        self.previous_signal = bool(signal)

        if live is not None:
            self.active, self.reason = bool(live), "gMain.inBattle"
        elif fresh_enemy:
            self.active, self.reason = True, "enemy RAM changed"
        elif moved:
            self.active, self.reason = False, "overworld movement"
        elif rising:
            self.active, self.reason = True, "battle signal"
        elif self.active:
            self.reason = "battle menu / animation"
        else:
            self.reason = "overworld / no new battle evidence"

        self.fingerprint = fingerprint
        if position is not None:
            self.position = position
        return int(self.active)
