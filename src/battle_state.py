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

    def update(self, party, position, flags=0, signal=False, live=None):
        fingerprint = enemy_fingerprint(party)
        fresh_enemy = bool(fingerprint and fingerprint != self.fingerprint)
        moved = position is not None and self.position is not None and position != self.position
        self.raw_flags = int(flags or 0)
        # A rising integration signal is supporting evidence, not a latch that
        # can permanently override actual walking after a fight.
        rising = bool(signal) and not self.previous_signal
        self.previous_signal = bool(signal)
        if live is not None:
            self.active, self.reason = bool(live), "gMain.inBattle"
        elif fresh_enemy:
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


class MainBattleReader:
    """Locate gMain by its ROM callbacks and live VBlank counter, not ROM version.

    pret/pokefirered include/main.h: counter +0x24, inBattle +0x439 bit 1.
    Validate two consecutive counter increments before trusting the bit.
    Unknown layouts return None so the conservative fallback remains available.
    """
    def __init__(self):
        self.candidates = None
        self.samples = 0
        self.offset = None

    def read(self, ram, frames=14):
        if ram is None or len(ram) < 0x48000:
            return None
        def u32(off):
            return int.from_bytes(bytes(ram[off:off+4]), 'little')
        def callback(value):
            return 0x08000001 <= value < 0x0A000000 and value & 1
        def valid(base):
            return callback(u32(base+4)) and callback(u32(base+12))
        if self.offset is not None:
            if valid(self.offset):
                return bool(int(ram[self.offset+0x439]) & 2)
            self.__init__()
            return None
        if self.candidates is None:
            self.candidates = {
                base: u32(base+0x24)
                for base in range(0x40000, 0x48000-0x43c, 4)
                if valid(base)
            }
            return None
        self.candidates = {
            base: u32(base+0x24) for base, previous in self.candidates.items()
            if valid(base) and (u32(base+0x24)-previous) % (2**32) == frames
        }
        self.samples += 1
        if not self.candidates:
            self.candidates = None
            self.samples = 0
        elif len(self.candidates) == 1 and self.samples >= 2:
            self.offset = next(iter(self.candidates))
            return bool(int(ram[self.offset+0x439]) & 2)
        return None
