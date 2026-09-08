import unittest

import battle_controller as ctrl
import battle_types as bt


def mon(species_types, *, slot=0, level=20, cur_hp=50, max_hp=50, status=0,
        atk=30, dfn=30, spd=30, spa=30, spdf=30, moves=()):
    return {
        "slot": slot, "checksum_ok": True, "level": level,
        "cur_hp": cur_hp, "max_hp": max_hp, "status": status,
        "types": list(species_types),
        "stats": {"attack": atk, "defense": dfn, "speed": spd,
                  "sp_attack": spa, "sp_defense": spdf},
        "stat_stages": {},
        "moves": list(moves),
    }


def mv(mtype, power, *, pp=10, acc=100, priority=0, is_status=False, mid=1):
    return {"id": mid, "type": mtype, "power": power, "pp": pp, "accuracy": acc,
            "priority": priority, "is_status": is_status, "mechanics_known": True}


def snap(player_active, enemy_active, *, player_party=None, is_trainer=False,
         can_escape=True):
    return {
        "player_active": player_active, "enemy_active": enemy_active,
        "player_party": player_party if player_party is not None else [player_active],
        "is_trainer": is_trainer, "can_escape": can_escape,
    }


class ControllerDecisionTests(unittest.TestCase):
    def test_fail_closed_when_no_active(self):
        d = ctrl.decide({"player_active": None, "enemy_active": None})
        self.assertEqual(d["action"], "MOVE_1")

    def test_fail_closed_when_enemy_unreadable(self):
        me = mon((bt.TYPE_WATER,), moves=[mv(bt.TYPE_NORMAL, 40, mid=1),
                                          mv(bt.TYPE_WATER, 40, mid=2)])
        d = ctrl.decide(snap(me, None))
        # attacks with a non-status move (index 0 or 1), never RUN/SWITCH blindly
        self.assertIn(d["action"], ("MOVE_1", "MOVE_2"))

    def test_picks_guaranteed_ko_move(self):
        me = mon((bt.TYPE_WATER,), level=30, spa=90,
                 moves=[mv(bt.TYPE_NORMAL, 20, mid=1),          # weak
                        mv(bt.TYPE_WATER, 60, mid=2)])          # STAB + SE
        enemy = mon((bt.TYPE_ROCK, bt.TYPE_GROUND), spdf=25, cur_hp=10, max_hp=40)
        d = ctrl.decide(snap(me, enemy))
        self.assertEqual(d["action"], "MOVE_2")
        self.assertIn("KO", d["reason"])

    def test_prefers_super_effective_for_expected_damage(self):
        me = mon((bt.TYPE_WATER,), level=20, spa=40,
                 moves=[mv(bt.TYPE_NORMAL, 40, mid=1),
                        mv(bt.TYPE_WATER, 40, mid=2)])          # SE vs Rock
        enemy = mon((bt.TYPE_ROCK,), spdf=40, cur_hp=120, max_hp=120)
        d = ctrl.decide(snap(me, enemy))
        self.assertEqual(d["action"], "MOVE_2")

    def test_switches_when_about_to_be_ko_and_safe_slot_exists(self):
        frail = mon((bt.TYPE_GRASS,), slot=0, cur_hp=6, max_hp=40, spd=10,
                    moves=[mv(bt.TYPE_GRASS, 40, mid=1)])
        wall = mon((bt.TYPE_WATER,), slot=1, cur_hp=60, max_hp=60, dfn=120,
                   spdf=120, moves=[mv(bt.TYPE_WATER, 40, mid=1)])
        enemy = mon((bt.TYPE_FIRE,), level=30, spa=90, spd=90,
                    moves=[mv(bt.TYPE_FIRE, 90, mid=1)])       # ~OHKOs the frail grass
        d = ctrl.decide(snap(frail, enemy, player_party=[frail, wall]))
        self.assertEqual(d["action"], "SWITCH_2")

    def test_no_switch_no_flee_in_trainer_battle_tries_best_trade(self):
        # trainer battle -> cannot flee, no safe switch -> must attack.
        frail = mon((bt.TYPE_GRASS,), slot=0, cur_hp=6, max_hp=40, spd=10, spa=40,
                    moves=[mv(bt.TYPE_GRASS, 60, mid=1),
                           mv(bt.TYPE_WATER, 60, mid=2)])
        alsofrail = mon((bt.TYPE_BUG,), slot=1, cur_hp=5, max_hp=30, dfn=10, spdf=10,
                        moves=[mv(bt.TYPE_BUG, 40, mid=1)])
        enemy = mon((bt.TYPE_FIRE,), level=40, spa=120, spd=120, spdf=20, cur_hp=40, max_hp=40,
                    moves=[mv(bt.TYPE_FIRE, 120, mid=1)])
        d = ctrl.decide(snap(frail, enemy, player_party=[frail, alsofrail],
                             is_trainer=True, can_escape=False))
        self.assertIn(d["action"], ("MOVE_1", "MOVE_2"))
        self.assertNotEqual(d["action"], "RUN")
        # Water hits the Fire enemy super-effectively -> that is the better trade
        self.assertEqual(d["action"], "MOVE_2")

    def test_runs_only_from_unwinnable_escapable_wild(self):
        weakling = mon((bt.TYPE_NORMAL,), level=5, atk=10, spd=10, cur_hp=8, max_hp=20,
                       moves=[mv(bt.TYPE_NORMAL, 40, mid=1)])
        # tank: huge bulk + a strong PHYSICAL attack so it OHKOs the weakling.
        tank = mon((bt.TYPE_ROCK, bt.TYPE_STEEL), level=40, atk=150, dfn=200, spdf=200,
                   spd=120, cur_hp=400, max_hp=400,
                   moves=[mv(bt.TYPE_ROCK, 100, mid=1)])
        d = ctrl.decide(snap(weakling, tank, can_escape=True, is_trainer=False))
        self.assertEqual(d["action"], "RUN")
        # ... but never from a trainer battle
        d2 = ctrl.decide(snap(weakling, tank, can_escape=False, is_trainer=True))
        self.assertNotEqual(d2["action"], "RUN")

    def test_status_only_moveset_uses_highest_pp_status_move(self):
        me = mon((bt.TYPE_NORMAL,),
                 moves=[mv(bt.TYPE_NORMAL, 0, is_status=True, pp=5, mid=1),
                        mv(bt.TYPE_NORMAL, 0, is_status=True, pp=30, mid=2)])
        enemy = mon((bt.TYPE_NORMAL,), cur_hp=50, max_hp=50)
        d = ctrl.decide(snap(me, enemy))
        self.assertEqual(d["action"], "MOVE_2")

    def test_unknown_move_mechanics_are_last_resort(self):
        me = mon((bt.TYPE_WATER,),
                 moves=[{"id": 999, "pp": 10, "mechanics_known": False},   # unknown
                        mv(bt.TYPE_WATER, 40, mid=2)])                     # known
        enemy = mon((bt.TYPE_ROCK,), spdf=40, cur_hp=120, max_hp=120)
        d = ctrl.decide(snap(me, enemy))
        self.assertEqual(d["action"], "MOVE_2")

    def test_out_of_pp_falls_back_safely(self):
        me = mon((bt.TYPE_WATER,), moves=[mv(bt.TYPE_WATER, 40, pp=0, mid=1),
                                          mv(bt.TYPE_NORMAL, 40, pp=0, mid=2)])
        d = ctrl.decide(snap(me, mon((bt.TYPE_NORMAL,))))
        self.assertEqual(d["action"], "MOVE_1")

    def test_almost_dead_slot_is_not_a_safe_switch(self):
        # frail active about to be KO'd; the only alternative is a 1/100-HP mon
        # with otherwise huge bulk. Using max_hp it would look "safe"; using
        # current HP it is not -> the controller must NOT switch into it.
        frail = mon((bt.TYPE_GRASS,), slot=0, cur_hp=6, max_hp=40, spd=10, spa=40,
                    moves=[mv(bt.TYPE_GRASS, 60, mid=1),
                           mv(bt.TYPE_FIRE, 60, mid=2)])
        chip = mon((bt.TYPE_WATER,), slot=1, cur_hp=1, max_hp=100, dfn=200,
                   spdf=200, moves=[mv(bt.TYPE_WATER, 40, mid=1)])
        enemy = mon((bt.TYPE_FIRE,), level=40, spa=120, spd=120, spdf=25,
                    cur_hp=200, max_hp=200, moves=[mv(bt.TYPE_FIRE, 110, mid=1)])
        d = ctrl.decide(snap(frail, enemy, player_party=[frail, chip],
                             is_trainer=True, can_escape=False))
        self.assertNotEqual(d["action"], "SWITCH_2")
        self.assertIn(d["action"], ("MOVE_1", "MOVE_2"))

    def test_decisions_are_not_executable_in_phase1(self):
        me = mon((bt.TYPE_WATER,), moves=[mv(bt.TYPE_WATER, 40, mid=1)])
        d = ctrl.decide(snap(me, mon((bt.TYPE_NORMAL,))))
        self.assertFalse(d["executable"])
        # an authoritative + cursor-confirmed snapshot would flip it
        s = snap(me, mon((bt.TYPE_NORMAL,)))
        s["active_slot_authoritative"] = True
        s["menu_cursor_known"] = True
        self.assertTrue(ctrl.decide(s)["executable"])
