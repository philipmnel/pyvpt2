# Library imports:
from qcelemental import constants as qcelconstants

wave_to_kcal = qcelconstants.conversion_factor("wavenumber", "kilocalorie per mol")
wave_to_kj = qcelconstants.conversion_factor("wavenumber", "kilojoule per mol")
wave_to_hartree = qcelconstants.get("inverse meter-hartree relationship") * 100
meter_to_bohr = qcelconstants.get("Bohr radius")
joule_to_hartree = qcelconstants.get("hartree-joule relationship")
mdyneA_to_hartreebohr = 100 * (meter_to_bohr ** 2) / (joule_to_hartree)
h = qcelconstants.get("Planck constant")
c = qcelconstants.get("speed of light in vacuum") * 100
kg_to_amu = qcelconstants.get("atomic mass constant")