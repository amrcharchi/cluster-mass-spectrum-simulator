import periodictable as pt
from itertools import product, combinations
import numpy as np



#=================================

def generate_clusters_general(elements, max_counts):

    if not 1 <= len(elements) <= 3:
        raise ValueError("Select between 1 and 3 elements.")

    if len(elements) != len(max_counts):
        raise ValueError("Each element needs a maximum atom count.")

    if len(set(elements)) != len(elements):
        raise ValueError("Elements must be different.")

    clusters = []

    for counts in product(
        *(range(n + 1) for n in max_counts)
    ):

        if sum(counts) == 0:
            continue

        clusters.append(counts)

    return clusters
#====================================


def get_element_isotopes(symbol):

    element = pt.elements.symbol(symbol)

    isotope_data = {}

    for mass_number in element.isotopes:

        isotope = element[mass_number]

        mass = isotope.mass
        abundance = isotope.abundance

        # Only isotopes with known and nonzero abundance
        if mass is not None and abundance is not None and abundance > 0:

            isotope_data[mass_number] = (mass, abundance)

    return isotope_data

#====================================

def isotope_pattern_general(composition, selected_isotopes):

    atom_options = []
    isotope_lookup = {}

    for element, n_atoms in composition.items():

        if n_atoms == 0:
            continue

        isotope_data = get_element_isotopes(element)
        chosen = selected_isotopes[element]

        if not chosen:
            raise ValueError(f"No isotopes selected for {element}")

        # Normalize abundances of the isotopes
        total_abundance = sum(isotope_data[n][1] for n in chosen)

        for n in chosen:
            mass, abundance = isotope_data[n]
            isotope_lookup[(element, n)] = (
                mass,
                abundance / total_abundance
            )

        # Each atom can have any of its selected isotopes
        for _ in range(n_atoms):
            atom_options.append([(element, n) for n in chosen])

    combined = {}

    for combination in product(*atom_options):

        key = tuple(sorted(combination))
        mass = 0.0
        probability = 1.0

        for isotope in combination:
            isotope_mass, isotope_probability = isotope_lookup[isotope]

            mass += isotope_mass
            probability *= isotope_probability

        if key not in combined:
            combined[key] = [mass, 0.0]

        combined[key][1] += probability

    return combined

#====================================

def find_cluster_overlaps_general(
    clusters, elements, selected_isotopes, tolerance=0.05
):

    all_patterns = {}

    # Calculate isotope patterns for all cluster compositions
    for cluster in clusters:

        composition = dict(zip(elements, cluster))

        all_patterns[cluster] = isotope_pattern_general(
            composition, selected_isotopes
        )

    overlapping_pairs = {}

    # Compare every pair of different clusters
    for cluster1, cluster2 in combinations(clusters, 2):

        matches = []

        pattern1 = all_patterns[cluster1]
        pattern2 = all_patterns[cluster2]

        for isotopes1, (mass1, prob1) in pattern1.items():
            for isotopes2, (mass2, prob2) in pattern2.items():

                difference = abs(mass1 - mass2)

                if difference < tolerance:

                    matches.append(
                        (isotopes1, isotopes2, mass1, mass2, difference)
                    )

        if matches:
            overlapping_pairs[(cluster1, cluster2)] = matches

    return overlapping_pairs


#======================================

def gaussian_spectrum_general(
    cluster, elements, selected_isotopes, mass_axis, resolving_power
):

    
    composition = dict(zip(elements, cluster))

    # Calculate isotope masses and probabilities
    pattern = isotope_pattern_general(
        composition, selected_isotopes
    )

    spectrum = np.zeros_like(mass_axis, dtype=float)

    for isotopes, (mass, probability) in pattern.items():

        # Gaussian width from mass resolving power
        fwhm = mass / resolving_power
        sigma = fwhm / 2.355

        # Gaussian peak profile
        gaussian = (
            1 / (sigma * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((mass_axis - mass) / sigma) ** 2)
        )

        spectrum += probability * gaussian

    return spectrum

