import numpy as np
import plotly.graph_objects as go
import streamlit as st
import periodictable as pt

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

    mass_regions = []

    for iso1, iso2, mass1, mass2, difference in matches:

        mass_regions.append(
            f"{min(mass1, mass2):.4f}–"
            f"{max(mass1, mass2):.4f}"
        )

    overlap_rows.append({
        "Cluster 1": cluster_name(cluster1, elements),
        "Cluster 2": cluster_name(cluster2, elements),
        "Peak pairs": len(matches),
        "Minimum Δm (Th)": round(
            min(match[4] for match in matches), 6
        ),
        "Mass intervals (Th)": ", ".join(mass_regions)
    })

st.write("Number of overlapping cluster pairs:", len(overlaps))

if overlap_rows:
    # Show a maximum of 6 rows before scrolling
    row_height = 35
    visible_rows = min(len(overlap_rows), 6)
    table_height = 40 + visible_rows * row_height

    # Match each table row to its original cluster pair
    overlap_pairs = list(overlaps.keys())

    # Refresh the table selection when the simulation changes
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
        selection_id = (table_key, row_index)

        if st.session_state.get("last_overlap_selection") != selection_id:

            # Select the two clusters for the graph
            st.session_state.plot_clusters = list(
                overlap_pairs[row_index]
            )

            st.session_state.last_overlap_selection = selection_id

    else:
        st.session_state.last_overlap_selection = None
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

if selected_clusters:

    # Calculate the mass points needed for all selected spectra
    mass_grids = []

    for cluster in selected_clusters:

        composition = dict(zip(elements, cluster))

        pattern = isotope_pattern_general(
            composition,
            selected_isotopes
        )

        for mass, probability in pattern.values():

            sigma = mass / (2.355 * R)

            mass_grids.append(
                np.linspace(
                    mass - 5 * sigma,
                    mass + 5 * sigma,
                    81
                )
            )

    mass_axis = np.unique(np.concatenate(mass_grids))

    # Create the figure
    fig = go.Figure()

    for cluster in selected_clusters:

        spectrum = gaussian_spectrum_general(
            cluster,
            elements,
            selected_isotopes,
            mass_axis,
            R
        )

        fig.add_trace(
            go.Scatter(
                x=mass_axis,
                y=spectrum,
                mode="lines",
                name=cluster_name(cluster, elements)
            )
        )

    # Calculate appropriate initial axis limits
    mass_min = float(np.min(mass_axis))
    mass_max = float(np.max(mass_axis))

    mass_span = mass_max - mass_min
    padding = max(0.3, 0.03 * mass_span)

    max_intensity = max(
        float(np.max(trace.y))
        for trace in fig.data
    )

    # Configure the figure
    fig.update_layout(
        height=680,
        xaxis_title="m/z (Th)",
        yaxis_title="Isotope probability density (1/Th)",
        hovermode="closest",
        dragmode="zoom",
        margin=dict(l=55, r=30, t=55, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    # Horizontal axis
    fig.update_xaxes(
        range=[
            max(0, mass_min - padding),
            mass_max + padding
        ],
        minallowed=0,
        fixedrange=False
    )

    # Vertical axis
    fig.update_yaxes(
        range=[0, max_intensity * 1.1],
        minallowed=0,
        fixedrange=False
    )

    # Display the interactive spectrum
    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "scrollZoom": False,
            "showAxisDragHandles": False,
            "modeBarButtonsToRemove": ["pan2d"]
        }
    )

else:
    st.info("Select at least one cluster to display its spectrum.")
