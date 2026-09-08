import unittest
import battle_types as bt
import battle_engine as be


def mon(level=20, types=(bt.TYPE_WATER,), atk=30, dfn=30, spd=30, spa=30, spdf=30,
        cur_hp=50, max_hp=50, status=0, stages=None):
    return {
        "level": level, "types": list(types), "cur_hp": cur_hp, "max_hp": max_hp,
        "status": status,
        "stats": {"attack": atk, "defense": dfn, "speed": spd,
                  "sp_attack": spa, "sp_defense": spdf},
        "stat_stages": stages or {},
    }


def move(mtype=bt.TYPE_WATER, power=40, accuracy=100, pp=25, priority=0,
         is_status=False):
    return {"type": mtype, "power": power, "accuracy": accuracy, "pp": pp,
            "priority": priority, "is_status": is_status}


class DamageEngineTests(unittest.TestCase):
    def test_status_move_is_not_a_damage_calc(self):
        d = be.damage_range(mon(), mon(), move(power=0, is_status=True))
        self.assertFalse(d["unknown"])         # not an error
        self.assertFalse(d["is_damaging"])
        self.assertIsNone(d["expected"])

    def test_unknown_move_type_is_fail_closed(self):
        d = be.damage_range(mon(), mon(), {"power": 40})   # no type
        self.assertTrue(d["unknown"])
        self.assertIsNone(d["min_damage"])

    def test_unknown_defender_types_is_fail_closed(self):
        atk = mon(types=(bt.TYPE_WATER,))
        dfn = mon(types=())     # no types
        d = be.damage_range(atk, dfn, move())
        self.assertTrue(d["unknown"])

    def test_immunity_gives_zero_not_unknown(self):
        atk = mon(types=(bt.TYPE_NORMAL,))
        dfn = mon(types=(bt.TYPE_GHOST,))
        d = be.damage_range(atk, dfn, move(mtype=bt.TYPE_NORMAL))
        self.assertFalse(d["unknown"])
        self.assertEqual(d["effectiveness"], 0.0)
        self.assertEqual(d["max_damage"], 0)

    def test_super_effective_beats_neutral_expected(self):
        atk = mon(level=25, types=(bt.TYPE_WATER,), spa=60)
        rock = mon(types=(bt.TYPE_ROCK,), spdf=40, max_hp=80, cur_hp=80)
        normal = mon(types=(bt.TYPE_NORMAL,), spdf=40, max_hp=80, cur_hp=80)
        se = be.damage_range(atk, rock, move(mtype=bt.TYPE_WATER, power=40))
        neu = be.damage_range(atk, normal, move(mtype=bt.TYPE_WATER, power=40))
        self.assertEqual(se["effectiveness"], 2.0)
        self.assertEqual(neu["effectiveness"], 1.0)
        self.assertGreater(se["expected"], neu["expected"])
        # STAB present (Water move, Water attacker)
        self.assertEqual(se["stab"], 1.5)

    def test_stab_increases_damage(self):
        watr = mon(types=(bt.TYPE_WATER,), spa=40)
        norm = mon(types=(bt.TYPE_NORMAL,), spa=40)
        d_stab = be.damage_range(watr, mon(types=(bt.TYPE_GRASS,), spdf=40),
                                 move(mtype=bt.TYPE_WATER))
        d_nost = be.damage_range(norm, mon(types=(bt.TYPE_GRASS,), spdf=40),
                                 move(mtype=bt.TYPE_WATER))
        self.assertEqual(d_stab["stab"], 1.5)
        self.assertEqual(d_nost["stab"], 1.0)
        self.assertGreater(d_stab["expected"], d_nost["expected"])

    def test_gen3_physical_uses_attack_special_uses_spatk(self):
        # a Ghost move is PHYSICAL in Gen III -> uses attack/defense
        atk = mon(types=(bt.TYPE_GHOST,), atk=100, spa=1)
        dfn = mon(types=(bt.TYPE_PSYCHIC,), dfn=50, spdf=1, max_hp=100, cur_hp=100)
        d = be.damage_range(atk, dfn, move(mtype=bt.TYPE_GHOST, power=40))
        self.assertTrue(d["physical"])
        self.assertFalse(d["unknown"])
        self.assertGreater(d["expected"], 0)
        # a Dark move is SPECIAL in Gen III -> uses spatk/spdef
        atk2 = mon(types=(bt.TYPE_DARK,), atk=1, spa=100)
        d2 = be.damage_range(atk2, dfn, move(mtype=bt.TYPE_DARK, power=40))
        self.assertFalse(d2["physical"])
        self.assertGreater(d2["expected"], 0)

    def test_burn_halves_physical_only(self):
        base = mon(types=(bt.TYPE_NORMAL,), atk=80)
        burned = mon(types=(bt.TYPE_NORMAL,), atk=80, status=be.STATUS_BURN)
        dfn = mon(types=(bt.TYPE_NORMAL,), dfn=40, max_hp=200, cur_hp=200)
        d_ok = be.damage_range(base, dfn, move(mtype=bt.TYPE_NORMAL, power=60))
        d_burn = be.damage_range(burned, dfn, move(mtype=bt.TYPE_NORMAL, power=60))
        self.assertLess(d_burn["expected"], d_ok["expected"])
        # special move unaffected by burn
        s_ok = be.damage_range(mon(types=(bt.TYPE_FIRE,), spa=80),
                               mon(types=(bt.TYPE_NORMAL,), spdf=40, max_hp=200, cur_hp=200),
                               move(mtype=bt.TYPE_FIRE, power=60))
        s_burn = be.damage_range(mon(types=(bt.TYPE_FIRE,), spa=80, status=be.STATUS_BURN),
                                 mon(types=(bt.TYPE_NORMAL,), spdf=40, max_hp=200, cur_hp=200),
                                 move(mtype=bt.TYPE_FIRE, power=60))
        self.assertAlmostEqual(s_ok["expected"], s_burn["expected"])

    def test_positive_attacker_stage_raises_damage(self):
        plain = mon(types=(bt.TYPE_NORMAL,), atk=50)
        boosted = mon(types=(bt.TYPE_NORMAL,), atk=50, stages={"attack": 2})
        dfn = mon(types=(bt.TYPE_NORMAL,), dfn=50, max_hp=300, cur_hp=300)
        d0 = be.damage_range(plain, dfn, move(mtype=bt.TYPE_NORMAL, power=60))
        d2 = be.damage_range(boosted, dfn, move(mtype=bt.TYPE_NORMAL, power=60))
        self.assertGreater(d2["expected"], d0["expected"] * 1.7)

    def test_ko_estimate(self):
        atk = mon(level=30, types=(bt.TYPE_WATER,), spa=80)
        weak = mon(types=(bt.TYPE_ROCK,), spdf=20, max_hp=12, cur_hp=12)
        tanky = mon(types=(bt.TYPE_GRASS,), spdf=200, max_hp=400, cur_hp=400)
        ko = be.ko_estimate(be.damage_range(atk, weak, move(mtype=bt.TYPE_WATER, power=60)), 12)
        self.assertTrue(ko["guaranteed"])
        no = be.ko_estimate(be.damage_range(atk, tanky, move(mtype=bt.TYPE_WATER, power=40)), 400)
        self.assertFalse(no["possible"])
        self.assertIsNotNone(no["fraction_expected"])
        # unknown -> all false
        u = be.ko_estimate({"unknown": True}, 50)
        self.assertFalse(u["guaranteed"] or u["possible"])

    def test_accuracy_and_speed_order(self):
        self.assertEqual(be.move_hits_probability(move(accuracy=70)), 0.7)
        self.assertIsNone(be.move_hits_probability({}))
        self.assertEqual(be.move_hits_probability(move(accuracy=0)), 1.0)   # "always hits"

        fast = mon(spd=100)
        slow = mon(spd=40)
        self.assertEqual(be.order_of_action(fast, move(), slow, move()), "a")
        self.assertEqual(be.order_of_action(slow, move(), fast, move()), "b")
        # priority beats speed
        self.assertEqual(be.order_of_action(slow, move(priority=1), fast, move()), "a")
        # unknown speed -> unknown
        self.assertEqual(be.order_of_action({"stats": {}}, move(), fast, move()),
                         "unknown")

    def test_paralysis_quarters_speed(self):
        self.assertAlmostEqual(be.effective_speed(mon(spd=100)), 100.0)
        self.assertAlmostEqual(
            be.effective_speed(mon(spd=100, status=be.STATUS_PARALYSIS)), 25.0)

    def test_crit_expected_is_separate_and_higher(self):
        atk = mon(level=25, types=(bt.TYPE_WATER,), spa=60)
        dfn = mon(types=(bt.TYPE_GRASS,), spdf=40, max_hp=80, cur_hp=80)
        d = be.damage_range(atk, dfn, move(mtype=bt.TYPE_WATER, power=40))
        self.assertIsNotNone(d["crit_expected"])
        self.assertGreater(d["crit_expected"], d["expected"])


class Gen3GoldenVectorTests(unittest.TestCase):
    """Fixed expected numbers taken by hand from the pret/pokefirered integer
    path. These MUST NOT re-implement the formula - they are literal constants.
    """

    def _phys(self, mtype, power, level, attack, defense,
              atk_types, def_types, **kw):
        a = {"level": level, "types": list(atk_types), "status": 0,
             "stats": {"attack": attack, "defense": defense, "speed": 1,
                       "sp_attack": attack, "sp_defense": defense},
             "stat_stages": {}}
        d = {"level": level, "types": list(def_types), "status": 0,
             "stats": {"attack": attack, "defense": defense, "speed": 1,
                       "sp_attack": attack, "sp_defense": defense},
             "stat_stages": {}}
        mv = {"type": mtype, "power": power, "accuracy": 100, "pp": 10,
              "priority": 0, "is_status": False}
        return be.calc_damage(a, d, mv, **kw)

    def test_mandatory_low_level_normal_stab_vector(self):
        # level=2, power=20, attack=5, defense=5, Normal STAB vs Normal.
        # FireRed integer path:
        #   levelTerm = 2*2//5 + 2                = 2
        #   base = 5*20*2 // 5 // 50 + 2          = 200//5//50 + 2 = 0 + 2 = 2
        #   STAB = 2 * 15 // 10                   = 3
        #   neutral type                          = 3
        #   rolls: 3*85//100 .. 3*100//100        = 2 .. 3
        r = self._phys(bt.TYPE_NORMAL, 20, 2, 5, 5,
                       (bt.TYPE_NORMAL,), (bt.TYPE_NORMAL,))
        self.assertEqual(r["min_damage"], 2)
        self.assertEqual(r["max_damage"], 3)
        self.assertEqual(len(r["rolls"]), 16)
        self.assertEqual(r["rolls"][0], 2)
        self.assertEqual(r["rolls"][-1], 3)
        self.assertEqual(set(r["rolls"]), {2, 3})
        self.assertEqual(r["stab"], 1.5)
        self.assertEqual(r["effectiveness"], 1.0)

    def test_no_stab_neutral_vector(self):
        # same numbers but attacker is Water (no STAB): base 2, no *1.5,
        # rolls 2*85//100 .. 2*100//100 = 1 .. 2
        r = self._phys(bt.TYPE_NORMAL, 20, 2, 5, 5,
                       (bt.TYPE_WATER,), (bt.TYPE_NORMAL,))
        self.assertEqual((r["min_damage"], r["max_damage"]), (1, 2))
        self.assertEqual(r["stab"], 1.0)

    def test_super_effective_doubles_after_stab_vector(self):
        # base 2 -> STAB 3 -> x2 type = 6 -> rolls 6*85//100 .. 6 = 5 .. 6
        r = self._phys(bt.TYPE_WATER, 20, 2, 5, 5,
                       (bt.TYPE_WATER,), (bt.TYPE_ROCK,))
        self.assertEqual(r["effectiveness"], 2.0)
        self.assertEqual((r["min_damage"], r["max_damage"]), (5, 6))

    def test_crit_doubles_base_before_stab_vector(self):
        # crit: base (2) *2 = 4 -> STAB 4*15//10 = 6 -> rolls 5..6
        r = self._phys(bt.TYPE_NORMAL, 20, 2, 5, 5,
                       (bt.TYPE_NORMAL,), (bt.TYPE_NORMAL,), is_crit=True)
        self.assertEqual((r["min_damage"], r["max_damage"]), (5, 6))

    def test_immunity_is_all_zero_rolls(self):
        r = self._phys(bt.TYPE_NORMAL, 20, 2, 5, 5,
                       (bt.TYPE_NORMAL,), (bt.TYPE_GHOST,))
        self.assertEqual(r["effectiveness"], 0.0)
        self.assertEqual(r["rolls"], [0] * 16)


class ConfidenceContractTests(unittest.TestCase):
    def _pair(self, **calc_kw):
        atk = mon(level=20, types=(bt.TYPE_WATER,), spa=40)
        dfn = mon(types=(bt.TYPE_NORMAL,), spdf=40, max_hp=200, cur_hp=200)
        return be.calc_damage(atk, dfn, move(mtype=bt.TYPE_WATER, power=60),
                              **calc_kw)

    def test_default_is_estimate_never_complete(self):
        d = self._pair()
        self.assertTrue(d["is_damaging"])
        self.assertFalse(d["mechanics_complete"])
        self.assertTrue(d["damage_is_estimate"])
        for r in d["unknown_reasons"]:
            self.assertIn(r, be.ALL_UNKNOWN_REASONS)
        self.assertIn(be.UNKNOWN_ABILITY, d["unknown_reasons"])
        self.assertIn(be.UNKNOWN_STAT_STAGES, d["unknown_reasons"])

    def test_all_known_marks_mechanics_complete(self):
        d = self._pair(ability_known=True, in_battle_types_authoritative=True,
                       stat_stages_known=True, weather_known=True,
                       screens_known=True, item_known=True)
        self.assertTrue(d["mechanics_complete"])
        self.assertFalse(d["damage_is_estimate"])
        self.assertEqual(d["unknown_reasons"], [])

    def test_dynamic_power_move_is_always_estimate(self):
        atk = mon(level=30, types=(bt.TYPE_NORMAL,), atk=60)
        dfn = mon(types=(bt.TYPE_NORMAL,), dfn=60, max_hp=200, cur_hp=200)
        seismic = {"type": bt.TYPE_FIGHTING, "power": 1, "accuracy": 100,
                   "pp": 10, "priority": 0, "is_status": False, "effect": 28}
        d = be.calc_damage(atk, dfn, seismic, ability_known=True,
                           in_battle_types_authoritative=True,
                           stat_stages_known=True, weather_known=True,
                           screens_known=True, item_known=True)
        self.assertFalse(d["mechanics_complete"])
        self.assertIn(be.UNKNOWN_DYNAMIC_POWER, d["unknown_reasons"])


class KoAssessmentTests(unittest.TestCase):
    def _lethal(self):
        atk = mon(level=40, types=(bt.TYPE_WATER,), spa=120)
        weak = mon(types=(bt.TYPE_ROCK,), spdf=20, max_hp=8, cur_hp=8)
        return be.damage_range(atk, weak, move(mtype=bt.TYPE_WATER, power=80)), 8

    def test_split_fields_present(self):
        dmg, hp = self._lethal()
        a = be.ko_assessment(dmg, hp, hit_probability=1.0)
        self.assertTrue(a["ko_on_hit_guaranteed"])
        self.assertTrue(a["ko_on_hit_possible"])
        self.assertEqual(a["ko_rolls_fraction"], 1.0)
        self.assertEqual(a["hit_probability"], 1.0)
        self.assertEqual(a["ko_probability"], 1.0)

    def test_true_guaranteed_needs_complete_mechanics_and_certain_hit(self):
        dmg, hp = self._lethal()
        # every roll KOs, but mechanics are an estimate -> NOT true_guaranteed
        self.assertFalse(be.ko_assessment(dmg, hp, hit_probability=1.0)["true_guaranteed"])
        # even with a full-mechanics damage calc, a sub-1.0 hit chance blocks it
        atk = mon(level=40, types=(bt.TYPE_WATER,), spa=120)
        weak = mon(types=(bt.TYPE_ROCK,), spdf=20, max_hp=8, cur_hp=8)
        full = be.calc_damage(atk, weak,
                              {"type": bt.TYPE_WATER, "power": 80, "accuracy": 90,
                               "pp": 10, "priority": 0, "is_status": False},
                              ability_known=True, in_battle_types_authoritative=True,
                              stat_stages_known=True, weather_known=True,
                              screens_known=True, item_known=True)
        self.assertTrue(full["mechanics_complete"])
        self.assertFalse(be.ko_assessment(full, 8, hit_probability=0.9)["true_guaranteed"])
        self.assertTrue(be.ko_assessment(full, 8, hit_probability=1.0)["true_guaranteed"])

    def test_unknown_ability_never_yields_true_guaranteed(self):
        # Levitate/Wonder Guard scenario: the estimate says "every roll KOs",
        # but because ability is not authoritative it must never be "true".
        dmg, hp = self._lethal()
        a = be.ko_assessment(dmg, hp, hit_probability=1.0)
        self.assertTrue(a["ko_on_hit_guaranteed"])
        self.assertFalse(a["true_guaranteed"])

    def test_ko_estimate_shim_guaranteed_is_on_hit_only(self):
        dmg, hp = self._lethal()
        k = be.ko_estimate(dmg, hp)
        self.assertTrue(k["guaranteed"])            # on-hit sense
        self.assertFalse(k["true_guaranteed"])      # strict sense
