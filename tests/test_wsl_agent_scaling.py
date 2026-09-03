import unittest

from pokemon_env import scaled_agent_slot


class WslAgentScalingTests(unittest.TestCase):
    def test_ten_agents_cover_the_120_agent_role_distribution(self):
        slots = [scaled_agent_slot(rank, 10) for rank in range(10)]
        self.assertEqual(slots, [0, 12, 24, 36, 48, 60, 72, 84, 96, 108])

    def test_single_agent_stays_in_first_role_slot(self):
        self.assertEqual(scaled_agent_slot(0, 1), 0)


if __name__ == "__main__":
    unittest.main()
