"""
MyActuator catalog (folder `my actuator/`).

The .xlsx and `motors_MyActuator_Series_L_H_X.csv` hold the same 30 motors (14 columns); the CSV is
read (no openpyxl needed). Format: ';' separator, decimal comma, UTF-8 BOM, 'N/A' for missing.
No file lists motor dimensions, so motors enter the dynamic model as point masses.
Note: the column `rated_voltage_A` is the rated CURRENT; it is renamed.
"""
import pandas as pd

from . import config as C


def load(path=C.CATALOG_CSV):
    df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig", na_values=["N/A"])
    df = df.rename(columns={"rated_voltage_A": "rated_current_A"})
    df["motor_name"] = df["motor_name"].str.strip()
    df["model"] = df["motor_name"].str.replace("MyActuator ", "", regex=False)
    # rated max speed: max_speed_rpm when given, otherwise the no-load speed
    df["speed_rpm"] = df["max_speed_rpm"].fillna(df["no_load_speed_rpm"])
    df["series"] = df["model"].str[0]
    required = ["peak_torque_Nm", "nominal_torque_Nm", "motor_mass_kg", "speed_rpm"]
    bad = df[df[required].isna().any(axis=1)]
    if len(bad):
        print(f"[catalog] dropping {len(bad)} motors with missing data: {list(bad['model'])}")
    df = df.dropna(subset=required).reset_index(drop=True)
    return df[["model", "series", "peak_torque_Nm", "nominal_torque_Nm", "speed_rpm", "motor_mass_kg",
               "rated_voltage_V", "rated_current_A", "stall_torque_Nm", "no_load_speed_rpm"]]
