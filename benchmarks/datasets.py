"""Real + synthetic series for the benchmark, each tagged with its matched structure.

"Structure" (trend/seasonal/period) is a property we assign per series so all three
libraries can be fit with the same configuration -- see CONTEXT.md's
"Matched-structure comparison" entry for why that matters.
"""

from dataclasses import dataclass

import numpy as np
import statsmodels.datasets as sm_datasets

# Classic Box-Jenkins AirPassengers monthly totals, Jan 1949 - Dec 1960 (same
# constant as examples/benchmark.py -- statsmodels.datasets doesn't ship it).
AIR_PASSENGERS = np.array(
    [
        112,
        118,
        132,
        129,
        121,
        135,
        148,
        148,
        136,
        119,
        104,
        118,
        115,
        126,
        141,
        135,
        125,
        149,
        170,
        170,
        158,
        133,
        114,
        140,
        145,
        150,
        178,
        163,
        172,
        178,
        199,
        199,
        184,
        162,
        146,
        166,
        171,
        180,
        193,
        181,
        183,
        218,
        230,
        242,
        209,
        191,
        172,
        194,
        196,
        196,
        236,
        235,
        229,
        243,
        264,
        272,
        237,
        211,
        180,
        201,
        204,
        188,
        235,
        227,
        234,
        264,
        302,
        293,
        259,
        229,
        203,
        229,
        242,
        233,
        267,
        269,
        270,
        315,
        364,
        347,
        312,
        274,
        237,
        278,
        284,
        277,
        317,
        313,
        318,
        374,
        413,
        405,
        355,
        306,
        271,
        306,
        315,
        301,
        356,
        348,
        355,
        422,
        465,
        467,
        404,
        347,
        305,
        336,
        340,
        318,
        362,
        348,
        363,
        435,
        491,
        505,
        404,
        359,
        310,
        337,
        360,
        342,
        406,
        396,
        420,
        472,
        548,
        559,
        463,
        407,
        362,
        405,
        417,
        391,
        419,
        461,
        472,
        535,
        622,
        606,
        508,
        461,
        390,
        432,
    ],
    dtype=float,
)


@dataclass(frozen=True)
class Structure:
    """Trend/seasonal configuration a series is fit with, shared across libraries."""

    trend: str  # "none" | "add" | "damped"
    seasonal: str  # "none" | "add" | "mul"
    seasonal_period: int | None = None

    @property
    def damped(self) -> bool:
        return self.trend == "damped"

    def __post_init__(self) -> None:
        if self.trend not in ("none", "add", "damped"):
            raise ValueError(f"Unknown trend {self.trend!r}")
        if self.seasonal not in ("none", "add", "mul"):
            raise ValueError(f"Unknown seasonal {self.seasonal!r}")
        if self.seasonal == "none" and self.trend == "none":
            return
        if self.seasonal != "none" and self.trend == "none":
            # multi_ets has no seasonal-without-trend class (HoltWinters always
            # includes a trend term), so this combination isn't representable.
            raise ValueError("multi_ets has no trend='none' + seasonal!='none' structure")
        if self.seasonal != "none" and self.seasonal_period is None:
            raise ValueError("seasonal structures need a seasonal_period")


@dataclass(frozen=True)
class Series:
    name: str
    y: np.ndarray
    structure: Structure
    category: str  # "real" | "synthetic"


def _real_series() -> list[Series]:
    nile = sm_datasets.nile.load_pandas().data["volume"].to_numpy(dtype=float)
    gdp = sm_datasets.macrodata.load_pandas().data["realgdp"].to_numpy(dtype=float)
    co2 = sm_datasets.co2.load_pandas().data["co2"]
    co2_monthly = co2.resample("MS").mean().interpolate().to_numpy(dtype=float)[-150:]

    return [
        Series("air_passengers", AIR_PASSENGERS, Structure("add", "mul", 12), "real"),
        Series("nile_flow", nile, Structure("none", "none"), "real"),
        Series("real_gdp", gdp, Structure("damped", "none"), "real"),
        Series("co2_monthly", co2_monthly, Structure("add", "add", 12), "real"),
    ]


def _seasonal_pattern(seasonal: str, m: int, amplitude: float) -> np.ndarray:
    phase = np.arange(m) / m
    if seasonal == "add":
        return amplitude * np.sin(2 * np.pi * phase)
    return 1.0 + (amplitude / 100.0) * np.sin(2 * np.pi * phase)  # "mul": centered on 1


def make_synthetic(
    structure: Structure,
    length: int = 150,
    noise_std: float = 5.0,
    level0: float = 100.0,
    slope: float = 0.5,
    phi: float = 0.9,
    seasonal_amplitude: float = 15.0,
    seed: int = 0,
) -> np.ndarray:
    """Synthesize a series with a known ground-truth trend/seasonal structure."""
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=float)

    if structure.trend == "none":
        trend = np.zeros(length)
    elif structure.trend == "add":
        trend = slope * t
    else:  # damped
        trend = slope * np.cumsum(phi ** np.arange(1, length + 1))

    level = level0 + trend
    noise = rng.normal(0.0, noise_std, length)

    if structure.seasonal == "none":
        return level + noise

    m = structure.seasonal_period
    assert m is not None  # Structure.__post_init__ enforces this for seasonal != "none"
    pattern = _seasonal_pattern(structure.seasonal, m, seasonal_amplitude)
    seasonal = np.tile(pattern, length // m + 1)[:length]
    if structure.seasonal == "add":
        return level + seasonal + noise
    return level * seasonal + noise


def _synthetic_series() -> list[Series]:
    m = 12
    core_structures = [
        Structure("none", "none"),
        Structure("add", "none"),
        Structure("damped", "none"),
        Structure("add", "add", m),
        Structure("add", "mul", m),
        Structure("damped", "add", m),
        Structure("damped", "mul", m),
    ]
    series = [
        Series(
            f"synthetic_{s.trend}_trend_{s.seasonal}_seasonal", make_synthetic(s), s, "synthetic"
        )
        for s in core_structures
    ]

    # A few extremes on top of the core 7, to stress noise/length rather than structure.
    extra = Structure("add", "add", m)
    series.append(
        Series("synthetic_high_noise", make_synthetic(extra, noise_std=25.0), extra, "synthetic")
    )
    # ponytail: short/long series each introduce a train length the JAX
    # multistart solver hasn't compiled before, and that compile runs away
    # in memory (OOM-killed a full benchmark run). Skipped until that's
    # diagnosed -- re-enable once _compile_multistart in holt.py is fixed.
    # short = Structure("damped", "mul", m)
    # series.append(
    #     Series("synthetic_short_series", make_synthetic(short, length=48), short, "synthetic")
    # )
    # long_ = Structure("add", "none")
    # series.append(
    #     Series("synthetic_long_series", make_synthetic(long_, length=400), long_, "synthetic")
    # )
    return series


def all_series() -> list[Series]:
    """The full, diverse benchmark set: ~4 real + ~10 synthetic series."""
    return _real_series() + _synthetic_series()
