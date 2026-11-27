import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location("make_split", Path(__file__).parents[1] / "scripts" / "make_split.py")
ms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ms)


def _ref(n):
    return pd.DataFrame({"bird": [f"b{i:03d}" for i in range(n)],
                         "sex": ["f" if i % 2 else "m" for i in range(n)],
                         "deploy_on": pd.to_datetime([f"{2024 + i % 3}-06-01" for i in range(n)], utc=True),
                         "tagged_young": [True if i % 5 == 0 else float("nan") for i in range(n)],
                         "migrant": [True if i % 7 == 0 else float("nan") for i in range(n)]})


def test_split_disjoint_proportional_deterministic():
    ref = _ref(92)
    birds = list(ref["bird"])
    search = birds[:12]
    exclude = set(birds[:24])
    s1 = ms.make_split(birds, ref, search, exclude, 20)
    s2 = ms.make_split(birds, ref, search, exclude, 20)
    assert s1 == s2
    assert len(s1["locked"]) == 20 and len(s1["search"]) == 12 and len(s1["train"]) == 60
    assert not set(s1["locked"]) & exclude
    assert not set(s1["train"]) & (set(s1["locked"]) | set(s1["search"]))
    sex = ref.set_index("bird").loc[s1["locked"], "sex"].value_counts()
    assert abs(sex.get("f", 0) - sex.get("m", 0)) <= 2
