import unittest

import numpy as np

from twoby2.nav_map import (NavCoordMap, NUM_CHANNELS, MAP_SIZE,
                            CH_EPISODE_VISITS, CH_GLOBAL_VISITS, CH_PLAYER,
                            CH_WARPS, CH_STRUCTURE)


class NavCoordMapTests(unittest.TestCase):
    def test_tensor_shape_and_dtype(self):
        m = NavCoordMap()
        m.observe((3, 0), 10, 10)
        t = m.tensor()
        self.assertEqual(t.shape, (NUM_CHANNELS, MAP_SIZE, MAP_SIZE))
        self.assertEqual(t.dtype, np.uint8)
        # player is always at the centre
        self.assertEqual(t[CH_PLAYER, MAP_SIZE // 2, MAP_SIZE // 2], 255)

    def test_coordinate_pays_once_per_episode_then_again_after_reset(self):
        m = NavCoordMap()
        self.assertTrue(m.observe((3, 0), 5, 5))     # first time -> reward
        self.assertFalse(m.observe((3, 0), 5, 5))    # same episode -> no reward
        m.reset_episode()
        self.assertTrue(m.observe((3, 0), 5, 5))     # after reset -> reward again

    def test_global_knowledge_survives_reset_but_does_not_block_episode_reward(self):
        m = NavCoordMap()
        m.observe((3, 0), 5, 5)
        m.reset_episode(keep_global=True)
        self.assertIn(((3, 0), 5, 5), m.global_visited)   # still known globally
        # episode channel is empty again -> the tile is "new" for this episode
        self.assertTrue(m.episode_tile_is_new((3, 0), 5, 5))

    def test_channels_are_centred_on_the_player(self):
        m = NavCoordMap()
        m.load_global(structure=[((3, 0), 12, 10)], warps=[((3, 0), 8, 10)])
        m.observe((3, 0), 10, 10)
        m.observe((3, 0), 11, 10)      # player now at (11, 10)
        t = m.tensor()
        half = MAP_SIZE // 2
        self.assertEqual(t[CH_STRUCTURE, half, half + 1], 180)   # (12,10) -> 1 east
        self.assertEqual(t[CH_WARPS, half, half - 3], 255)       # (8,10)  -> 3 west
        self.assertEqual(t[CH_EPISODE_VISITS, half, half - 1], 255)  # (10,10) -> 1 west

    def test_tiles_from_other_maps_are_not_drawn(self):
        m = NavCoordMap()
        m.load_global(visited=[((9, 9), 10, 10)])
        m.observe((3, 0), 10, 10)
        t = m.tensor()
        # only the player's own tile on this map is plotted in the global channel
        self.assertEqual(int((t[CH_GLOBAL_VISITS] > 0).sum()), 1)


if __name__ == "__main__":
    unittest.main()
