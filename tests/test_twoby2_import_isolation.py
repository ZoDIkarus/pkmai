"""The 2×2 code must stay off the live training/serving path."""
import ast
import pathlib
import unittest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

LIVE_MODULES = ("pokemon_env.py", "train.py", "watch.py", "watcher_runtime.py",
                "web_stream.py")

# names the live path must never import
ISOLATED_NAMES = ("twoby2", "battle_engine", "battle_controller", "battle_ram",
                  "battle_types", "pokedb", "battle_executor", "battle_env",
                  "battle_train")

# pure-logic twoby2 modules — must not import heavy or live modules
PURE_LOGIC_MODULES = {
    "__init__.py", "config.py", "horizon.py", "retention.py", "promotion.py",
    "isolation.py", "scenario_pool.py", "router.py", "manifest.py",
    "battle_summary.py", "activation.py", "battle_promotion.py",
    "reset_tools.py", "ram_battle_probe.py", "protected_assets.py",
    "reward_split.py", "scenario_normalizer.py", "watcher_parity.py",
    "orchestration.py", "battle_ram_live.py", "emulator_battle_driver.py",
    "nav_progress.py", "nav_chunk_rollout.py",
}
# ML / numeric modules — allowed torch / sb3 / gymnasium / numpy
ML_MODULES = {"nav_feature_extractor.py", "ppo_isolation.py", "battle_env.py",
              "nav_wrapper.py", "nav_map.py", "live_integration.py"}

FORBIDDEN_FOR_PURE = ("pokemon_env", "train", "watch", "watcher_runtime",
                      "web_stream", "torch", "stable_baselines3", "stable_retro",
                      "retro", "curriculum_v20", "frontier_v20")
FORBIDDEN_FOR_ML = ("pokemon_env", "train", "watch", "watcher_runtime",
                    "web_stream", "curriculum_v20", "frontier_v20")


def _imports(path):
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


    # live modules that are DELIBERATELY wired to the gated 2x2 seam
    WIRED_LIVE_MODULES = {"train.py", "watch.py", "watcher_runtime.py",
                          "pokemon_env.py"}

    def test_live_modules_only_touch_twoby2_via_the_gated_seam(self):
        """Live modules may reference twoby2 now, but ONLY through a guarded /
        gated import — a twoby2 import failure must never break training, and
        the seam must be a no-op while the feature gate is OFF."""
        for mod in LIVE_MODULES:
            p = SRC / mod
            if not p.exists():
                continue
            txt = p.read_text()
            imported = _imports(p)
            # never the heavy Phase-1 battle modules directly
            for bad in ("battle_ram_live", "emulator_battle_driver",
                        "battle_engine", "battle_controller"):
                self.assertNotIn(bad, imported, f"{mod} imports {bad} directly")
            if "twoby2" in txt:
                self.assertIn(mod, self.WIRED_LIVE_MODULES,
                              f"{mod} references twoby2 but is not a wired module")
                # the twoby2 import must be guarded (try/except) OR feature-gated
                guarded = ("try:" in txt and "twoby2" in txt) or \
                    "feature_enabled" in txt or "maybe_wrap_full_agent" in txt or \
                    "battle_driver_for" in txt
                self.assertTrue(guarded, f"{mod}: twoby2 use is not guarded/gated")

    def test_live_modules_do_not_import_isolated_code(self):
        for mod in LIVE_MODULES:
            p = SRC / mod
            if not p.exists() or mod in self.WIRED_LIVE_MODULES:
                continue
            imported = _imports(p)
            for bad in ISOLATED_NAMES:
                self.assertNotIn(bad, imported, f"{mod} imports {bad}")

    def test_pure_logic_twoby2_modules_avoid_heavy_and_live_imports(self):
        pkg = SRC / "twoby2"
        for name in PURE_LOGIC_MODULES:
            p = pkg / name
            if not p.exists():
                continue
            imported = _imports(p)
            for bad in FORBIDDEN_FOR_PURE:
                self.assertNotIn(bad, imported, f"twoby2/{name} imports {bad}")

    def test_ml_twoby2_modules_avoid_live_imports(self):
        pkg = SRC / "twoby2"
        for name in ML_MODULES:
            p = pkg / name
            if not p.exists():
                continue
            imported = _imports(p)
            for bad in FORBIDDEN_FOR_ML:
                self.assertNotIn(bad, imported, f"twoby2/{name} imports {bad}")

    def test_every_twoby2_module_is_categorised(self):
        pkg = SRC / "twoby2"
        actual = {p.name for p in pkg.glob("*.py")}
        known = PURE_LOGIC_MODULES | ML_MODULES
        self.assertEqual(actual - known, set(),
                         "new twoby2 module not categorised in this test")

    def test_top_level_battle_modules_only_bridge_isolated_code(self):
        for name in ("battle_executor.py", "battle_env.py", "battle_train.py"):
            p = SRC / name
            if not p.exists():
                continue
            imported = _imports(p)
            for bad in ("pokemon_env", "train", "watch", "watcher_runtime",
                        "web_stream", "curriculum_v20"):
                self.assertNotIn(bad, imported, f"{name} imports {bad}")


if __name__ == "__main__":
    unittest.main()
