#
# Copyright 2024 Dan J. Bower
#
# This file is part of Thermofit.
#
# Thermofit is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Thermofit is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Thermofit. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Thermofit"""

# Convenient to use chemical symbol names so pylint: disable=invalid-name

from __future__ import annotations

import logging
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from thermochem.janaf import Janafdb, JanafPhase

logger: logging.Logger = logging.getLogger(__name__)

ZERO_TEMPERATURE: float = 1e-2
"""To avoid NaNs with log fitting perturb zero temperatures slightly away from 0 K"""

db: Janafdb = Janafdb()


class FunctionLibrary:
    """Library of fitting functions

    T is temperature and a, b, c, d, ... are fitting coefficients
    """

    @staticmethod
    def linear_function(T, a, b) -> npt.NDArray[np.float64]:
        return a + b * T

    @staticmethod
    def quadratic_function(T, a, b, c) -> npt.NDArray[np.float64]:
        return a + b * T + c * T**2

    @staticmethod
    def cubic_function(T, a, b, c, d) -> npt.NDArray[np.float64]:
        return a + b * T + c * T**2 + d * T**3

    @staticmethod
    def quartic_function(T, a, b, c, d, e) -> npt.NDArray[np.float64]:
        return a + b * T + c * T**2 + d * T**3 + e * T**4

    @staticmethod
    def hoff_kitchoff(T, a, b, c, d, e) -> npt.NDArray[np.float64]:
        """Equation 2.9 in Stock et al. (2018), FastChem"""
        return a / T + b * np.log(T) + c + d * T + e * T**2


class FitDeltafG:
    """Fits the delta Gibbs energy of formation in the online JANAF tables.

    Args:
        formula: Chemical formula
        phase: Phase
        replacements: Option to replace incorrect values in the JANAF tables. Defaults to None.
        fitting_function. Function to fit. Defaults to `hoff_kirchoff`.

    Attributes:
        formula: Chemical formula
        phase: Phase
        phase_data: Thermochem phase data
        rawdata: Raw data from Thermochem
        data: Processed data
        T: Temperature array
        Delta_fG: Delta Gibbs energy of formation array
        fitting_function: Fitting function
    """

    def __init__(
        self,
        formula: str,
        phase: str,
        replacements: dict[float, float] | None = None,
        fitting_function: Callable = FunctionLibrary.hoff_kitchoff,
    ):
        self.formula: str = formula
        self.phase: str = phase
        self.phase_data: JanafPhase = db.getphasedata(formula, phase=phase)
        self.rawdata: pd.DataFrame = self.phase_data.rawdata
        self.data: pd.DataFrame = self.rawdata.copy(deep=True)
        self.data = self.data.dropna(subset=["Delta_fG"])
        if replacements:
            self.replacements: dict[float, float] = replacements
            self.data["Delta_fG"] = self.data.apply(self.replace_gibbs, axis=1)
        self.data["T"] = self.data["T"].replace(0, ZERO_TEMPERATURE)

        self.T: npt.NDArray[np.float64] = self.data["T"].to_numpy()
        self.Delta_fG: npt.NDArray[np.float64] = self.data["Delta_fG"].to_numpy()
        self.fitting_function = fitting_function

    @property
    def name(self) -> str:
        """Descriptive name"""
        return f"{self.formula}_{self.phase}"

    def replace_gibbs(self, row: pd.Series) -> float:
        """Replace values"""
        return self.replacements.get(row["T"], row["Delta_fG"])

    def fit_custom_function(
        self, T: npt.NDArray[np.float64], Delta_fG: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Fits the function

        Args:
            T: temperature
            Delta_fG: Delta Gibbs energy of formation

        Returns:
            Fitting coefficients
        """
        popt, _ = curve_fit(self.fitting_function, T, Delta_fG)

        # Predict using the fitted function
        Delta_fG_pred = self.fitting_function(T, *popt)

        # Calculate R-squared value
        r2 = np.array(r2_score(Delta_fG, Delta_fG_pred))

        return popt, r2, Delta_fG_pred

    def fit(self, show: bool = False) -> None:
        """Fits the data using the chosen functional form and plots the result.

        Args:
            show: Shows the plot
        """
        popt, r2, Delta_fG_pred = self.fit_custom_function(self.T, self.Delta_fG)

        # Print R-squared value and fit parameters
        logger.info("%s R-squared value: %f", self.name, r2)
        logger.info("%s Fit parameters: %s", self.name, popt)
        popt_str: str = ", ".join([f"{x:.6e}" for x in popt])
        # [f"{parameter}, " for parameter in popt]
        logger.info("%s Fit parameters (copy-paste): %s", self.name, "[" + popt_str + "]")

        _, ax = plt.subplots()

        # Plotting
        ax.scatter(self.T, self.Delta_fG, color="blue", label="Data")
        ax.plot(
            self.T, Delta_fG_pred, color="red", label=f"Fit with {self.fitting_function.__name__}"
        )
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Delta fG (kJ/mol)")
        ax.set_title(f"Fit of Delta fG for {self.name} with {self.fitting_function.__name__}")
        ax.legend()

        if show:
            plt.show()


# C_cr is reference
CH4_g: FitDeltafG = FitDeltafG("CH4", "g")
# Cl2 is reference
CO_g: FitDeltafG = FitDeltafG("CO", "g")
CO2_g: FitDeltafG = FitDeltafG("CO2", "g")
# H2_g is reference
H2O_g: FitDeltafG = FitDeltafG("H2O", "g")
# Replacement fixes a sign error in JANAF
H2O_l: FitDeltafG = FitDeltafG("H2O", "l", replacements={380: -224.102})
# FIXME: Bad fit
# H2S_g: FitDeltafG = FitDeltafG("H2S", "g")
# N2_g is reference
NH3_g: FitDeltafG = FitDeltafG("H3N", "g")
# O2_g is reference
# FIXME: Bad fit
# S2_g: FitDeltafG = FitDeltafG("S2", "g")
# FIXME: Bad fit
# SO_g: FitDeltafG = FitDeltafG("OS", "g")
# FIXME: Bad fit
# SO2_g: FitDeltafG = FitDeltafG("O2S", "g")


all_data_fits: dict[str, FitDeltafG] = {
    "CH4_g": CH4_g,
    "CO": CO_g,
    "CO2": CO2_g,
    "H2O_g": H2O_g,
    "H2O_l": H2O_l,
    # "H2S_g": H2S_g,
    "NH3_g": NH3_g,
    # "S2_g": S2_g,
    # "SO_g": SO_g,
    # "SO2_g": SO2_g,
}
