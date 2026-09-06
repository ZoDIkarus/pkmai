"""Verify the real reward selection and the shared battle scale, without training."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from pokemon_env import PokemonFireRedEnv as Env


class FighterRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(Path('src/pokemon_env.py').read_text())
        cls.selection = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
            and any(isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'reward' for t in x.targets)
                    and isinstance(x.value, ast.Call) and isinstance(x.value.func, ast.Name) and x.value.func.id == 'sum'
                    for x in n.body))
        cls.code = compile(ast.Module(body=[cls.selection], type_ignores=[]), '<real fighter reward>', 'exec')

    def select(self, mode, general, combat):
        scope = dict(self=SimpleNamespace(training_mode=mode), reward=general,
                     reward_events=['new_stage:+250','fighter_leash:truncate'], combat_rewards=combat)
        exec(self.code,scope)
        return scope['reward'],scope['reward_events']

    def test_exploration_and_story_never_pay_fighter(self):
        reward,events=self.select('FIGHTER',1300,[])
        self.assertEqual(reward,0)
        self.assertEqual(events,['fighter_leash:truncate'])

    def test_exact_combat_amounts_not_rounded_log_values(self):
        combat=[('battle_win',10),('enemy_damage',.08*7),('took_damage',-.1*3),('battle_step',-.005)]
        reward,events=self.select('FIGHTER',300,combat)
        self.assertAlmostEqual(reward,10.255)
        self.assertFalse(any('new_stage' in e for e in events))
        self.assertEqual(self.select('FIGHTER',250,[('party_wiped',-100)])[0],-100)

    def test_other_modes_keep_existing_total(self):
        for mode in ('FULL','BRIDGE','FRONTIER','RETENTION'):
            self.assertEqual(self.select(mode,321,[('battle_win',10)])[0],321)

    def test_only_fighter_ignores_three_faint_decay(self):
        e=Env.__new__(Env)
        e._is_trainer_battle=lambda:False
        e.post_wipe_recovery=False
        bank,map_id=next(iter(e.WILD_TRAINING_MAPS))
        for count in (0,2,3,20):
            e.episode_wild_faints=count
            e.training_mode='FIGHTER'
            self.assertEqual(e._battle_reward_scale(bank,map_id),1)
            e.training_mode='FULL'
            self.assertEqual(e._battle_reward_scale(bank,map_id),1 if count<3 else .1)
        e._is_trainer_battle=lambda:True
        for mode in ('FIGHTER','FULL'):
            e.training_mode=mode
            self.assertEqual(e._battle_reward_scale(bank,map_id),2)

    def test_post_wipe_scale_and_exemptions_stay_shared(self):
        e=Env.__new__(Env);e._is_trainer_battle=lambda:False
        e.post_wipe_recovery=True;e.episode_wild_faints=0
        bank,map_id=next(iter(e.WILD_TRAINING_MAPS))
        for mode in ('FIGHTER','FULL'):
            e.training_mode=mode
            self.assertEqual(e._battle_reward_scale(bank,map_id),.05)
            self.assertEqual(e._battle_reward_scale(bank,map_id,post_wipe_exempt=True),1)

class FrontierResetTests(unittest.TestCase):
    def env(self,mode='FRONTIER'):
        e=Env.__new__(Env);e.training_mode=mode;e.episode_start_stage=2
        return e

    def test_frontier_cannot_spend_episode_back_in_town(self):
        e=self.env()
        for _ in range(e.FRONTIER_BACKTRACK_STEPS-1):
            self.assertFalse(e._frontier_backtrack_expired(1,True,False))
        self.assertTrue(e._frontier_backtrack_expired(1,True,False))
        self.assertFalse(e._frontier_backtrack_expired(2,True,False))
        self.assertEqual(e._frontier_backtrack_steps,0)

    def test_invalid_reads_and_battles_do_not_count(self):
        e=self.env()
        for _ in range(200):
            self.assertFalse(e._frontier_backtrack_expired(1,False,False))
            self.assertFalse(e._frontier_backtrack_expired(1,True,True))
        self.assertEqual(getattr(e,'_frontier_backtrack_steps',0),0)

    def test_full_bridge_retention_and_fighter_not_leashed_by_frontier(self):
        for mode in ('FULL','BRIDGE','RETENTION','FIGHTER'):
            e=self.env(mode)
            for _ in range(200):
                self.assertFalse(e._frontier_backtrack_expired(1,True,False))
