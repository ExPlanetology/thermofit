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
"""This script will generate plots of the data fits and output the fit coefficients"""

import logging
from logging import Logger

import matplotlib.pyplot as plt

from thermofit import debug_logger
from thermofit.thermofit import all_data_fits

logger: Logger = debug_logger()
logger.setLevel(logging.INFO)

for name, data in all_data_fits.items():
    data.fit(show=False)

# all_data_fits["H2S_g"].fit(lower_temperature=0, upper_temperature=800)

plt.show()
