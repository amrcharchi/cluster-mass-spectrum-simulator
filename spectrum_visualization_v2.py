import numpy as np
import plotly.graph_objects as go
import streamlit as st

from spectrum_model_v2 import isotope_pattern_general, gaussian_spectrum_general


def render_spectrum_visualization(
    selected_clusters, elements, selected_isotopes, R, cluster_name
):
    """Render individual and hypothetical-mixture mass spectra.
    cluster_name is the existing formatter: cluster_name(cluster, elements).
    """
    spectrum_mode = st.radio(
        "Spectrum display mode",
        ["Individual spectra", "Hypothetical mixture"],
        horizontal=True,
        key="spectrum_display_mode",
    )

    if not selected_clusters:
        st.info("Select at least one cluster to display a spectrum.")

    else:
        # Compute the original probability-density spectra on one common mass axis.
        mass_grids = []
        peak_masses = []

        for cluster in selected_clusters:
            composition = dict(zip(elements, cluster))
            pattern = isotope_pattern_general(composition, selected_isotopes)

            for mass, probability in pattern.values():
                sigma = mass / (2.355 * R)
                peak_masses.append(mass)
                mass_grids.append(
                    np.linspace(max(0.0, mass - 5 * sigma), mass + 5 * sigma, 81)
                )

        mass_axis = np.unique(np.concatenate(mass_grids))

        densities = {
            cluster: gaussian_spectrum_general(
                cluster, elements, selected_isotopes, mass_axis, R
            )
            for cluster in selected_clusters
        }

        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd",
            "#d62728", "#17becf", "#8c564b", "#e377c2",
        ]
        fig = go.Figure()

        if spectrum_mode == "Individual spectra":
            y_display = st.radio(
                "Vertical axis",
                ["Relative intensity (%)", "Probability density (1/Th)"],
                horizontal=True,
                key="individual_spectrum_y_axis",
            )

            for i, cluster in enumerate(selected_clusters):
                density = densities[cluster]
                if y_display == "Relative intensity (%)":
                    maximum = np.max(density)
                    values = 100 * density / maximum if maximum > 0 else density
                else:
                    values = density

                fig.add_trace(go.Scatter(
                    x=mass_axis,
                    y=values,
                    mode="lines",
                    name=cluster_name(cluster, elements),
                    line=dict(color=colors[i % len(colors)], width=2),
                ))

            y_axis_title = y_display
            st.caption(
                "Each cluster is displayed independently. In relative-intensity mode, "
                "the strongest peak of each cluster is scaled to 100%; this does not "
                "represent its abundance relative to other clusters."
            )

        else:  # Hypothetical mixture
            st.caption(
                "Specify assumed relative detected signal contributions. "
                "These are not predictions of how many clusters the experiment produces."
            )

            
            # Equal initial contribution for every selected cluster
            n_clusters = len(selected_clusters)
            default_percentage = 100.0 / n_clusters

            # Detect when the cluster selection changes
            current_selection = tuple(selected_clusters)

            selection_changed = (
                st.session_state.get("previous_mixture_clusters")
                != current_selection
            )

            if selection_changed:
                st.session_state["previous_mixture_clusters"] = current_selection

            percentages = {}

            for cluster in selected_clusters:
                name = cluster_name(cluster, elements)
                input_key = f"mixture_percent_{name}"

                # Reset ALL contributions when clusters are added or removed
                if selection_changed or input_key not in st.session_state:
                    st.session_state[input_key] = default_percentage

                col1, col2 = st.columns([3, 1])

                with col1:
                    st.write(name)

                with col2:
                    percentages[cluster] = st.number_input(
                        f"Contribution (%) for {name}",
                        min_value=0.0,
                        max_value=100.0,
                        step=1.0,
                        format="%g",
                        key=input_key,
                        label_visibility="collapsed",
                    )

            # Check that entered percentages add up to 100%
            percentage_sum = sum(percentages.values())

            st.write(f"**Total contribution: {percentage_sum:.1f}%**")

            if not np.isclose(percentage_sum, 100.0, atol=1e-6):
                st.warning(
                    "The contributions must add up to 100%. "
                    "Please adjust the numbers."
                )

            else:
                fractions = {
                    cluster: percentage / 100.0
                    for cluster, percentage in percentages.items()
                }


                show_components = st.checkbox(
                    "Show individual weighted contributions",
                    value=True,
                    key="show_mixture_components",
                )

                components = {
                    cluster: fractions[cluster] * densities[cluster]
                    for cluster in selected_clusters
                }
                total_density = sum(
                    components.values(), np.zeros_like(mass_axis)
                )

                # Exactly one common normalization: components must not be
                # individually normalized before or after their weighted sum.
                peak = np.max(total_density)
                scale = 100.0 / peak if peak > 0 else 0.0

                if show_components:
                    for i, cluster in enumerate(selected_clusters):
                        fig.add_trace(go.Scatter(
                            x=mass_axis,
                            y=scale * components[cluster],
                            mode="lines",
                            name=(
                                f"{cluster_name(cluster, elements)} "
                                f"({100 * fractions[cluster]:.1f}%)"
                            ),
                            line=dict(
                                color=colors[i % len(colors)],
                                width=1.8,
                                dash="dash",
                            ),
                        ))

                fig.add_trace(go.Scatter(
                    x=mass_axis,
                    y=scale * total_density,
                    mode="lines",
                    name="Hypothetical total",
                    line=dict(color="#202020", width=3),
                ))

                y_axis_title = "Relative intensity (%)"
                st.caption(
                    "The mixture is formed from the original probability-density "
                    "spectra and normalized to a 100% maximum only afterward. "
                )

        
        if fig.data:
            minimum_mass = min(peak_masses)
            maximum_mass = max(peak_masses)
            plot_padding = max(
                0.02,
                0.02 * (maximum_mass - minimum_mass),
                3 * maximum_mass / (2.355 * R),
            )
            plot_max = max(float(np.max(trace.y)) for trace in fig.data)

            fig.update_layout(
                template="plotly_white",
                height=680,
                xaxis_title="m/z (Th)",
                yaxis_title=y_axis_title,
                dragmode="zoom",
                hovermode="x unified",
                margin=dict(l=35, r=25, t=30, b=50),
                legend=dict(orientation="h", y=1.02, x=0),
            )
            fig.update_xaxes(
                range=[max(0.0, minimum_mass - plot_padding), maximum_mass + plot_padding],
                minallowed=0,
                fixedrange=False,
            )
            fig.update_yaxes(
                range=[0, 1.1 * plot_max if plot_max > 0 else 1],
                minallowed=0,
                fixedrange=False,
            )

            st.plotly_chart(
                fig,
                width="stretch",
                config={
                    "displayModeBar": True,
                    "scrollZoom": False,
                    "showAxisDragHandles": False,
                    
                },
            )
