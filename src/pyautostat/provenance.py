"""Stable local content references and optional DataFrame fingerprints."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, time, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import InvalidDataError
from .specifications import _json_value


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_reference(kind: str, value: Any) -> str:
    """Reference a JSON snapshot; mapping keys sort, array order is preserved."""
    if not kind or not kind.isascii() or not kind.replace("_", "").isalnum():
        raise InvalidDataError("Content reference kind must be an ASCII identifier.")
    digest = hashlib.sha256(kind.encode("ascii") + b"\0" + _canonical(value)).hexdigest()
    return f"sha256:{kind}:{digest}"


def _scalar(value: Any) -> list[Any]:
    """Encode supported values with explicit types; never stringify unknown objects."""
    if isinstance(value, tuple):
        return ["tuple", [_scalar(item) for item in value]]
    if value is None or value is pd.NA or value is pd.NaT:
        return ["missing"]
    if isinstance(value, (np.datetime64, pd.Timestamp)):
        if pd.isna(value):
            return ["missing"]
        return ["datetime", pd.Timestamp(value).isoformat()]
    if isinstance(value, (np.timedelta64, pd.Timedelta)):
        if pd.isna(value):
            return ["missing"]
        return ["timedelta_ns", int(pd.Timedelta(value).value)]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        return ["missing"] if math.isnan(value) else ["float", value.hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, datetime):
        return ["datetime", value.isoformat()]
    if isinstance(value, date):
        return ["date", value.isoformat()]
    if isinstance(value, time):
        return ["time", value.isoformat()]
    if isinstance(value, timedelta):
        return ["timedelta_us", value.total_seconds() * 1_000_000]
    raise InvalidDataError(
        f"Dataset fingerprint cannot encode {type(value).__name__}; "
        "convert unsupported object values to a declared, stable dtype first."
    )


def dataset_fingerprint(frame: pd.DataFrame) -> dict[str, Any]:
    """Hash schema, index and values in row/column order without retaining observations."""
    if not isinstance(frame, pd.DataFrame):
        raise InvalidDataError("Dataset fingerprint requires a pandas DataFrame.")
    if any(not isinstance(name, str) for name in frame.columns):
        raise InvalidDataError("Dataset fingerprint requires string column names.")
    columns: list[dict[str, Any]] = []
    for name, series in frame.items():
        descriptor: dict[str, Any] = {"name": name, "dtype": str(series.dtype)}
        if isinstance(series.dtype, pd.CategoricalDtype):
            descriptor["categories"] = [_scalar(item) for item in series.cat.categories]
            descriptor["ordered"] = bool(series.cat.ordered)
        columns.append(descriptor)
    header = {
        "algorithm": "pyautostat-dataframe-sha256-v1",
        "columns": columns,
        "index_type": type(frame.index).__name__,
        "index_dtype": str(frame.index.dtype),
        "index_names": [_scalar(name) for name in frame.index.names],
        "rows": len(frame),
    }
    hasher = hashlib.sha256()
    hasher.update(_canonical(header) + b"\n")
    try:
        for index, row in zip(frame.index, frame.itertuples(index=False, name=None), strict=True):
            hasher.update(_canonical([_scalar(index), [_scalar(item) for item in row]]) + b"\n")
    except InvalidDataError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidDataError(f"Dataset fingerprint could not encode the data: {exc}") from exc
    return {"algorithm": header["algorithm"], "digest": hasher.hexdigest()}
