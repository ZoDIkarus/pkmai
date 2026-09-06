"""Once-per-trainer episode bonuses, separate from ordinary combat shaping."""
class TrainerRewards:
    def __init__(self, start=50.0, win=50.0, brock_start=500.0):
        self.start_value, self.win_value, self.brock_value = start, win, brock_start
        self.reset()

    def reset(self):
        self.started = set()
        self.won = set()
        self.active_id = None
        self.saw_ongoing = False

    def update(self, active, is_trainer, trainer_id, outcome):
        events = []
        if active and is_trainer and self.active_id is None and 0 < trainer_id < 743:
            self.active_id = trainer_id
            self.saw_ongoing = False
            if trainer_id not in self.started:
                self.started.add(trainer_id)
                value = self.brock_value if trainer_id == 414 else self.start_value
                events.append(('brock_battle_start' if trainer_id == 414 else 'trainer_battle_start', value))
        if self.active_id is not None:
            if active and outcome == 0:
                self.saw_ongoing = True
            # Outcome 1 is the complete battle win, not EXP from one enemy KO.
            if self.saw_ongoing and outcome == 1 and self.active_id not in self.won:
                self.won.add(self.active_id)
                events.append(('trainer_battle_won', self.win_value))
            if not active:
                self.active_id = None
                self.saw_ongoing = False
        return events
