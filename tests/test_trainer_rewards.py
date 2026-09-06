import unittest
from trainer_rewards import TrainerRewards
from firered_ram import read_trainer_battle, read_battle_type_flags
from types import SimpleNamespace

class TrainerBonusTests(unittest.TestCase):
    def test_complete_win_pays_once_and_enemy_ko_does_not(self):
        r=TrainerRewards()
        self.assertEqual(r.update(True,True,12,0),[('trainer_battle_start',50)])
        for _ in range(5):self.assertEqual(r.update(True,True,12,0),[])
        self.assertEqual(r.update(True,True,12,1),[('trainer_battle_won',50)])
        self.assertEqual(r.update(False,False,12,1),[])
        self.assertEqual(r.update(True,True,12,0),[])
        self.assertEqual(r.update(False,False,12,1),[])

    def test_repeated_loss_cannot_farm_start_and_does_not_pay_win(self):
        r=TrainerRewards()
        self.assertEqual(r.update(True,True,12,0),[('trainer_battle_start',50)])
        self.assertEqual(r.update(False,False,12,2),[])
        self.assertEqual(r.update(True,True,12,0),[])
        self.assertEqual(r.update(False,False,12,2),[])
        self.assertEqual(r.update(True,True,13,0),[('trainer_battle_start',50)])

    def test_brock_uses_500_start_not_550(self):
        r=TrainerRewards()
        self.assertEqual(r.update(True,True,414,0),[('brock_battle_start',500)])
        self.assertEqual(r.update(False,False,414,1),[('trainer_battle_won',50)])
        r.reset()
        self.assertEqual(r.update(True,True,414,0),[('brock_battle_start',500)])

    def test_wild_invalid_id_and_stale_outcome_do_not_pay_win(self):
        r=TrainerRewards()
        self.assertEqual(r.update(True,False,12,0),[])
        self.assertEqual(r.update(True,True,0,0),[])
        self.assertEqual(r.update(True,True,12,1),[('trainer_battle_start',50)])
        self.assertEqual(r.update(False,False,12,1),[])

    def test_reader_uses_verified_addresses(self):
        ram=bytearray(0x40000)
        ram[0x22b4c:0x22b50]=(8).to_bytes(4,'little')
        ram[0x386ae:0x386b0]=(414).to_bytes(2,'little')
        ram[0x23e8a]=1
        env=SimpleNamespace(get_ram=lambda:ram)
        self.assertEqual(read_battle_type_flags(env),8)
        self.assertEqual(read_trainer_battle(env),(414,1))
        self.assertEqual(read_trainer_battle(SimpleNamespace(get_ram=lambda:b'')),(0,None))
