"""Conservative battle tracking for FireRed's persistent enemy-party RAM.

Battle type flags are not a live-battle boolean (ordinary wild battles can
have zero flags). Enemy changes start a battle; stationary menus keep it
active. Confirmed overworld movement ends it even if old enemy data remains.
"""
def enemy_fingerprint(party):
    return tuple((int(m.get('slot', -1)), int(m.get('species_id', 0)),
                  int(m.get('personality', 0)), int(m.get('cur_hp', 0)),
                  tuple((int(a.get('id', 0)), int(a.get('pp', 0)))
                        for a in m.get('moves', []))) for m in party or [])


class BattleState:
    def __init__(self, party=(), position=None):
        self.fingerprint = enemy_fingerprint(party)
        self.position = position
        self.active = False
        self.raw_flags = 0
        self.reason = 'reset'
        self.previous_signal = False

    def update(self, party, position, flags=0, signal=False):
        fingerprint = enemy_fingerprint(party)
        fresh_enemy = bool(fingerprint and fingerprint != self.fingerprint)
        moved = position is not None and self.position is not None and position != self.position
        self.raw_flags = int(flags or 0)
        # A rising integration signal is supporting evidence, not a latch that
        # can permanently override actual walking after a fight.
        rising = bool(signal) and not self.previous_signal
        self.previous_signal = bool(signal)
        if fresh_enemy:
            self.active, self.reason = True, 'enemy RAM changed'
        elif moved:
            self.active, self.reason = False, 'overworld movement'
        elif rising:
            self.active, self.reason = True, 'battle signal'
        elif self.active:
            self.reason = 'battle menu / animation'
        else:
            self.reason = 'overworld / no new battle evidence'
        self.fingerprint = fingerprint
        if position is not None:
            self.position = position
        return int(self.active)
