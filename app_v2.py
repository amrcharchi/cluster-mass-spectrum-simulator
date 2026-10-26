import numpy as np
import plotly.graph_objects as go
import streamlit as st
from collections import Counter
import periodictable as pt
from spectrum_visualization_v2 import render_spectrum_visualization


from spectrum_model_v2 import (
    generate_clusters_general,
    get_element_isotopes,
    isotope_pattern_general,
    find_cluster_overlaps_general,
    gaussian_spectrum_general
)


def cluster_name(cluster, elements):

    return "".join(
        f"{element}{n if n > 1 else ''}"
        for element, n in zip(elements, cluster)
        if n > 0
    ) + "+"


st.set_page_config(
    page_title="Cluster Mass Spectrum Simulator",
    layout="wide"
)



def group_overlap_regions(matches):

    sorted_matches = sorted(
        matches,
        key=lambda match: min(match[2], match[3])
    )

    regions = []

    for match in sorted_matches:

        mass1 = match[2]
        mass2 = match[3]

        start = min(mass1, mass2)
        end = max(mass1, mass2)

        if not regions or start > regions[-1]["end"]:

            regions.append({
                "start": start,
                "end": end,
                "matches": [match]
            })

        else:

            regions[-1]["end"] = max(
                regions[-1]["end"], end
            )

            regions[-1]["matches"].append(match)

    return regions


def isotopologue_label(isotopes):

    counts = Counter(isotopes)
    parts = []

    for (element, mass_number), number in sorted(counts.items()):

        label = f"{mass_number}{element}"

        if number > 1:
            label += f"×{number}"

        parts.append(label)

    return " · ".join(parts)


st.title("Cluster Mass Spectrum Simulator")

# 1. Cluster generation

st.header("1. Cluster generation")

# Elements with available natural isotope abundance data
available_elements = [
    element.symbol
    for element in pt.elements
    if element.number > 0
    and get_element_isotopes(element.symbol)
]


# Shared selection for search and periodic table
if "chosen_elements" not in st.session_state:
    st.session_state.chosen_elements = []


def toggle_element(symbol):

    selected = st.session_state.chosen_elements.copy()

    if symbol in selected:
        selected.remove(symbol)

    elif len(selected) < 3:
        selected.append(symbol)

    st.session_state.chosen_elements = selected


# Option 1: Search box
st.multiselect(
    "Search for elements:",
    options=available_elements,
    max_selections=3,
    key="chosen_elements",
    placeholder="Type an element symbol..."
)


# Option 2: Periodic table
with st.expander("Or choose elements from the periodic table"):

    st.caption(
        "Click to select or deselect an element. "
        "Selected elements are highlighted."
    )

    periodic_rows = [
        ["H"] + [""] * 16 + ["He"],

        ["Li", "Be"] + [""] * 10
        + "B C N O F Ne".split(),

        ["Na", "Mg"] + [""] * 10
        + "Al Si P S Cl Ar".split(),

        "K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr".split(),

        "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe".split(),

        "Cs Ba Ln Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn".split(),

        "Fr Ra An Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split(),

        [""] * 2
        + "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split()
        + [""],

        [""] * 2
        + "Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split()
        + [""]
    ]

    for row_index, row in enumerate(periodic_rows):

        if row_index == 7:
            st.caption("Lanthanides")

        if row_index == 8:
            st.caption("Actinides")

        columns = st.columns(18, gap="small")

        for i, symbol in enumerate(row):

            if not symbol:
                continue

            if symbol in ("Ln", "An"):
                columns[i].caption(symbol)
                continue

            selected = symbol in st.session_state.chosen_elements

            disabled = (
                symbol not in available_elements
                or (
                    len(st.session_state.chosen_elements) >= 3
                    and not selected
                )
            )

            columns[i].button(
                symbol,
                key=f"pt_{symbol}",
                on_click=toggle_element,
                args=(symbol,),
                type="primary" if selected else "secondary",
                disabled=disabled,
                width="stretch"
            )

    st.caption(
        "Greyed-out elements have no available natural isotope "
        "distribution in the current model, or the selection is full."
    )


elements = st.session_state.chosen_elements

if not elements:
    st.info("Select up to three elements to begin.")
    st.stop()


st.subheader("Maximum atom counts")

max_counts = []

columns = st.columns(len(elements))

for column, element in zip(columns, elements):
    with column:
        count = st.number_input(
        f"Maximum {element}",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
        placeholder="Enter maximum",
        key=f"max_{element}"
    )
        max_counts.append(count)

if any(count is None for count in max_counts):
    st.info("Enter the maximum number of atoms for each selected element.")
    st.stop()

clusters = generate_clusters_general(elements, max_counts)

st.write("Number of possible clusters:", len(clusters))




# 2. Isotope selection

st.header("2. Isotope selection")

if "isotope_table_version" not in st.session_state:
    st.session_state.isotope_table_version = 0


# Called when a user changes an isotope dropdown
def isotope_dropdown_changed():
    st.session_state.isotope_table_version += 1


# Called when a user checks or unchecks an isotope in the table
def isotope_table_changed(editor_key, row_lookup):

    edits = st.session_state[editor_key]["edited_rows"]

    for row_index, changes in edits.items():

        if "Included" not in changes:
            continue

        element, mass_number = row_lookup[int(row_index)]
        widget_key = f"isotopes_{element}"

        chosen = set(st.session_state[widget_key])

        if changes["Included"]:
            chosen.add(mass_number)
        else:
            chosen.discard(mass_number)

        st.session_state[widget_key] = sorted(chosen)

    # Refresh the table with the updated selections
    st.session_state.isotope_table_version += 1



selected_isotopes = {}
isotope_rows = []
row_lookup = []

for element in elements:

    isotope_data = get_element_isotopes(element)
    available_isotopes = sorted(isotope_data.keys())

    widget_key = f"isotopes_{element}"

    # Default: only 16O for oxygen, all isotopes otherwise
    if widget_key not in st.session_state:

        if element == "O" and 16 in available_isotopes:
            st.session_state[widget_key] = [16]
        else:
            st.session_state[widget_key] = available_isotopes

    chosen = st.multiselect(
        f"Select {element} isotopes:",
        options=available_isotopes,
        format_func=lambda n, e=element: f"{n}{e}",
        key=widget_key,
        on_change=isotope_dropdown_changed
    )

    selected_isotopes[element] = chosen

    for mass_number, (mass, abundance) in isotope_data.items():

        isotope_rows.append({
            "Element": element,
            "Isotope": f"{mass_number}{element}",
            "Exact mass (u)": mass,
            "Natural abundance (%)": abundance,
            "Included": mass_number in chosen
        })

        row_lookup.append((element, mass_number))


st.subheader("Isotope information")

editor_key = (
    f"isotope_editor_{'_'.join(elements)}_"
    f"{st.session_state.isotope_table_version}"
)

st.data_editor(
    isotope_rows,
    key=editor_key,
    width="stretch",
    hide_index=True,
    num_rows="fixed",
    disabled=[
        "Element",
        "Isotope",
        "Exact mass (u)",
        "Natural abundance (%)"
    ],
    column_config={
        "Exact mass (u)": st.column_config.NumberColumn(
            format="%.6f"
        ),
        "Natural abundance (%)": st.column_config.NumberColumn(
            format="%.4f"
        ),
        "Included": st.column_config.CheckboxColumn(
            "Included",
            help="Include this isotope in the simulation"
        )
    },
    on_change=isotope_table_changed,
    args=(editor_key, row_lookup)
)

# Require at least one isotope for each selected element
for element in elements:
    if not selected_isotopes[element]:
        st.warning(f"Select at least one isotope of {element}.")
        st.stop()





# 3. Potential mass overlaps

# Store the clusters selected for plotting
if "plot_clusters" not in st.session_state:
    st.session_state.plot_clusters = []

# Remove selections that are no longer valid
st.session_state.plot_clusters = [
    cluster
    for cluster in st.session_state.plot_clusters
    if cluster in clusters
]

st.header("3. Potential mass overlaps")

tolerance = st.number_input(
    "Mass proximity threshold (Th)",
    min_value=0.001,
    max_value=0.500,
    value=0.050,
    step=0.010,
    format="%.3f"
)

# Prevent excessively large calculations
estimated_combinations = 1

for element, maximum in zip(elements, max_counts):
    n_isotopes = len(selected_isotopes[element])
    estimated_combinations *= sum(
        n_isotopes ** n for n in range(maximum + 1)
    )

if len(clusters) > 150 or estimated_combinations > 5000:
    st.warning(
        "This selection generates too many cluster or isotope "
        "combinations. Please reduce the maximum atom counts."
    )
    st.stop()

overlaps = find_cluster_overlaps_general(
    clusters,
    elements,
    selected_isotopes,
    tolerance
)

overlap_rows = []

for (cluster1, cluster2), matches in overlaps.items():

    regions = group_overlap_regions(matches)

    overlap_rows.append({
        "Cluster 1": cluster_name(cluster1, elements),
        "Cluster 2": cluster_name(cluster2, elements),
        "Peak-pair matches": len(matches),
        "Overlap regions": len(regions),
        "Minimum Δm (Th)": round(
            min(match[4] for match in matches), 6
        )
    })


st.write("Number of overlapping cluster pairs:", len(overlaps))


if overlap_rows:

    # Show up to six table rows before scrolling
    row_height = 35
    visible_rows = min(len(overlap_rows), 6)
    table_height = 40 + visible_rows * row_height

    overlap_pairs = list(overlaps.keys())

    table_key = (
        f"overlap_{elements}_{max_counts}_"
        f"{selected_isotopes}_{tolerance}"
    )

    table_event = st.dataframe(
        overlap_rows,
        width="stretch",
        hide_index=True,
        height=table_height,
        row_height=row_height,
        on_select="rerun",
        selection_mode="single-row",
        key=table_key
    )

    selected_rows = table_event.selection.rows

    if selected_rows:

        row_index = selected_rows[0]
        selected_pair = overlap_pairs[row_index]

        # Preserve the existing click-to-plot behavior
        selection_id = (table_key, row_index)

        if st.session_state.get("last_overlap_selection") != selection_id:

            st.session_state.plot_clusters = list(selected_pair)
            st.session_state.last_overlap_selection = selection_id

        cluster1, cluster2 = selected_pair

        # Group the matches belonging to this pair
        regions = group_overlap_regions(
            overlaps[selected_pair]
        )

        with st.expander("Isotopologue details"):

            region_index = st.selectbox(
                "Choose a mass region:",
                options=list(range(len(regions))),
                format_func=lambda i: (
                    f"Region {i + 1}: "
                    f"{regions[i]['start']:.4f}–"
                    f"{regions[i]['end']:.4f} Th "
                    f"({len(regions[i]['matches'])} matches)"
                ),
                key=f"region_{elements}_{selected_pair}"
            )

            region = regions[region_index]

            # Obtain the probabilities of each isotopologue
            pattern1 = isotope_pattern_general(
                dict(zip(elements, cluster1)),
                selected_isotopes
            )

            pattern2 = isotope_pattern_general(
                dict(zip(elements, cluster2)),
                selected_isotopes
            )

            detail_rows = []

            for iso1, iso2, mass1, mass2, difference in region["matches"]:

                prob1 = pattern1[iso1][1]
                prob2 = pattern2[iso2][1]

                detail_rows.append({
                    "Isotopologue 1": isotopologue_label(iso1),
                    "Isotopologue 2": isotopologue_label(iso2),
                    "Mass 1 (Th)": round(mass1, 6),
                    "Mass 2 (Th)": round(mass2, 6),
                    "Δm (Th)": round(difference, 6),
                    "P1 (%)": f"{100 * prob1:.6g}",
                    "P2 (%)": f"{100 * prob2:.6g}"
                })

            st.dataframe(
                detail_rows,
                width="stretch",
                hide_index=True,
                row_height=35,
                height=40 + 35 * min(len(detail_rows), 6)
            )

            st.caption(
                "Probabilities are relative to each cluster's "
                "selected isotope distribution. Mass regions are "
                "grouped using nearby theoretical peak positions, "
                "not instrumental resolution."
            )

    else:
        st.session_state.last_overlap_selection = None
        st.caption(
            "Select a cluster pair to display its spectra "
            "and inspect the contributing isotopologues."
        )

else:
    st.info("No nearby isotope peaks were found.")





# 4. Spectrum visualization

st.header("4. Mass spectrum visualization")

selected_clusters = st.multiselect(
    "Choose cluster spectra to display:",
    options=clusters,
    format_func=lambda cluster: cluster_name(cluster, elements),
    placeholder="Select one or more clusters...",
    key="plot_clusters"
)

R = st.slider(
    "Mass resolving power (R = m / FWHM)",
    min_value=500,
    max_value=10000,
    value=3000,
    step=100
)

render_spectrum_visualization(
    selected_clusters=selected_clusters,
    elements=elements,
    selected_isotopes=selected_isotopes,
    R=R,
    cluster_name=cluster_name
)