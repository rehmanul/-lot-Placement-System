"""
Enterprise-Grade Îlot Placement System
Specialized for architectural îlot placement with size distributions
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import tempfile
import os
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any
import json

# Import our specialized modules
from src.dwg_parser import DWGParser
from src.ilot_placement_engine import IlotPlacementEngine
from src.zone_classifier import ZoneClassifier

# Page config
st.set_page_config(
    page_title="🏗️ Enterprise Îlot Placement System", 
    page_icon="🏗️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'zones' not in st.session_state:
    st.session_state.zones = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'file_loaded' not in st.session_state:
    st.session_state.file_loaded = False

def main():
    st.title("🏗️ Enterprise Îlot Placement System")
    st.markdown("**Professional îlot placement with size distribution control and automatic corridor generation**")

    # Sidebar for file upload and parameters
    with st.sidebar:
        st.header("📁 File Upload")
        uploaded_file = st.file_uploader(
            "Upload DXF/DWG Plan", 
            type=['dxf', 'dwg'],
            help="Upload your architectural plan in DXF or DWG format"
        )

        if uploaded_file:
            load_architectural_plan(uploaded_file)

        if st.session_state.file_loaded:
            st.success(f"✅ Plan loaded: {len(st.session_state.zones)} zones detected")

            # Îlot Distribution Parameters
            st.header("📊 Îlot Size Distribution")
            st.markdown("Configure the percentage distribution of îlot sizes:")

            col1, col2 = st.columns(2)
            with col1:
                dist_0_1 = st.slider("0-1 m²", 0, 50, 10, help="Percentage of îlots between 0-1 m²")
                dist_3_5 = st.slider("3-5 m²", 0, 50, 30, help="Percentage of îlots between 3-5 m²")
            with col2:
                dist_1_3 = st.slider("1-3 m²", 0, 50, 25, help="Percentage of îlots between 1-3 m²")
                dist_5_10 = st.slider("5-10 m²", 0, 50, 35, help="Percentage of îlots between 5-10 m²")

            # Validate distribution totals 100%
            total_dist = dist_0_1 + dist_1_3 + dist_3_5 + dist_5_10
            if total_dist != 100:
                st.warning(f"⚠️ Distribution total: {total_dist}% (should be 100%)")
                if st.button("Auto-normalize to 100%"):
                    factor = 100 / total_dist
                    dist_0_1 = int(dist_0_1 * factor)
                    dist_1_3 = int(dist_1_3 * factor)
                    dist_3_5 = int(dist_3_5 * factor)
                    dist_5_10 = 100 - dist_0_1 - dist_1_3 - dist_3_5
                    st.rerun()

            # Advanced Parameters
            st.header("⚙️ Advanced Settings")
            corridor_width = st.slider("Corridor Width (m)", 0.8, 2.0, 1.2, 0.1)
            total_area = st.number_input("Total Available Area (m²)", 100, 10000, 1000, 50)

            # Analysis Button
            if st.button("🚀 Generate Îlot Placement", type="primary"):
                run_ilot_analysis(
                    distribution={'0-1': dist_0_1, '1-3': dist_1_3, '3-5': dist_3_5, '5-10': dist_5_10},
                    total_area=total_area,
                    corridor_width=corridor_width
                )

    # Main content area
    if not st.session_state.file_loaded:
        show_welcome_screen()
    else:
        show_analysis_interface()

def show_welcome_screen():
    """Display welcome screen with instructions"""
    st.markdown("""
    ## 🎯 Welcome to the Enterprise Îlot Placement System

    This professional tool automatically places îlots in architectural plans with:

    ### ✨ Key Features:
    - **Size Distribution Control**: Define exact percentages for different îlot sizes
    - **Automatic Zone Detection**: Identifies walls, entrances, and restricted areas
    - **Corridor Generation**: Automatically creates corridors between facing îlot rows
    - **Constraint Compliance**: Respects entrance/exit zones and restricted areas
    - **Enterprise Visualization**: Professional color-coded output

    ### 📋 Getting Started:
    1. Upload your DXF/DWG architectural plan using the sidebar
    2. Configure your îlot size distribution (percentages)
    3. Set corridor width and total area parameters
    4. Click "Generate Îlot Placement" to see results

    ### 🎨 Zone Color Legend:
    - **Black**: Walls (îlots can touch)
    - **Red**: Entrances/Exits (no îlots allowed)
    - **Light Blue**: Restricted areas (stairs, elevators)
    - **White**: Available space for îlots
    """)

def load_architectural_plan(uploaded_file):
    """Load and parse architectural plan"""
    try:
        with st.spinner("📖 Loading architectural plan..."):
            # Parse DWG/DXF file
            parser = DWGParser()
            file_bytes = uploaded_file.read()

            zones = parser.parse_file_simple(file_bytes, uploaded_file.name)

            if not zones:
                st.error("❌ No zones detected in the uploaded file")
                return

            # Classify zones for îlot placement
            classifier = ZoneClassifier()
            classified_zones = classifier.classify_zones(zones)

            st.session_state.zones = classified_zones
            st.session_state.file_loaded = True

            st.success(f"✅ Successfully loaded {len(zones)} zones from {uploaded_file.name}")

    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.info("Please ensure the file is a valid DXF or DWG format")

def run_ilot_analysis(distribution: Dict[str, float], total_area: float, corridor_width: float):
    """Run îlot placement analysis"""
    try:
        with st.spinner("🤖 Generating îlot placement..."):
            # Initialize îlot placement engine
            engine = IlotPlacementEngine()

            # Run placement analysis
            results = engine.place_ilots_with_distribution(
                zones=st.session_state.zones,
                distribution=distribution,
                total_area=total_area,
                corridor_width=corridor_width
            )

            st.session_state.analysis_results = results

            if results.get('error'):
                st.error(f"❌ Analysis failed: {results['error']}")
            else:
                st.success("✅ Îlot placement generated successfully!")

    except Exception as e:
        st.error(f"❌ Analysis error: {str(e)}")

def show_analysis_interface():
    """Display analysis results and visualization"""
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Visualization", "📊 Statistics", "🔍 Details", "📄 Export"])

    with tab1:
        show_visualization()

    with tab2:
        show_statistics()

    with tab3:
        show_detailed_results()

    with tab4:
        show_export_options()

def show_visualization():
    """Display the main visualization"""
    st.subheader("🗺️ Îlot Placement Visualization")

    if not st.session_state.analysis_results:
        st.info("🎯 Run îlot placement analysis to see visualization")
        return

    # Visualization controls
    col1, col2, col3 = st.columns(3)
    with col1:
        show_zones = st.checkbox("Show Zones", True)
    with col2:
        show_ilots = st.checkbox("Show Îlots", True)
    with col3:
        show_corridors = st.checkbox("Show Corridors", True)

    # Create interactive plot
    fig = create_ilot_visualization(show_zones, show_ilots, show_corridors)
    st.plotly_chart(fig, use_container_width=True)

    # Zone legend
    if show_zones:
        st.subheader("🎨 Zone Legend")
        classifier = ZoneClassifier()
        legend = classifier.create_zone_legend()

        legend_cols = st.columns(len(legend))
        for i, (zone_type, info) in enumerate(legend.items()):
            with legend_cols[i]:
                st.markdown(f"""
                <div style="background-color: {info['color']}; padding: 10px; border-radius: 5px; text-align: center; margin: 5px;">
                    <strong>{info['label']}</strong><br>
                    <small>{info['description']}</small>
                </div>
                """, unsafe_allow_html=True)

def create_ilot_visualization(show_zones: bool, show_ilots: bool, show_corridors: bool):
    """Create the main îlot placement visualization"""
    fig = go.Figure()

    # Add zones
    if show_zones:
        for i, zone in enumerate(st.session_state.zones):
            points = zone.get('points', [])
            if len(points) >= 3:
                x_coords = [p[0] for p in points] + [points[0][0]]
                y_coords = [p[1] for p in points] + [points[0][1]]

                zone_type = zone.get('zone_type', 'AVAILABLE')
                classifier = ZoneClassifier()
                color = classifier._get_zone_color(zone_type)

                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    fill="toself",
                    mode='lines',
                    name=f'{zone_type}' if i == 0 else None,
                    showlegend=(i == 0),
                    fillcolor=color,
                    line=dict(color='black', width=1),
                    hovertemplate=f"<b>Zone {i}</b><br>Type: {zone_type}<br>Area: {zone.get('area', 0):.1f} m²<extra></extra>"
                ))

    # Add îlots
    if show_ilots and 'placed_ilots' in st.session_state.analysis_results:
        ilots = st.session_state.analysis_results['placed_ilots']

        # Color by size range
        size_colors = {
            '0-1': 'rgba(255, 200, 0, 0.8)',    # Yellow
            '1-3': 'rgba(0, 255, 0, 0.8)',      # Green
            '3-5': 'rgba(0, 150, 255, 0.8)',    # Blue
            '5-10': 'rgba(255, 0, 150, 0.8)'    # Magenta
        }

        for ilot in ilots:
            bounds = ilot['bounds']
            x_coords = [bounds[0], bounds[2], bounds[2], bounds[0], bounds[0]]
            y_coords = [bounds[1], bounds[1], bounds[3], bounds[3], bounds[1]]

            size_range = ilot.get('size_range', 'unknown')
            color = size_colors.get(size_range, 'rgba(128, 128, 128, 0.8)')

            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                fill="toself",
                mode='lines',
                name=f'Îlots {size_range} m²' if size_range not in [trace.name for trace in fig.data],
                fillcolor=color,
                line=dict(color='darkred', width=2),
                hovertemplate=f"<b>Îlot {ilot['id']}</b><br>Size: {size_range} m²<br>Area: {ilot['area']:.2f} m²<extra></extra>"
            ))

    # Add corridors
    if show_corridors and 'corridors' in st.session_state.analysis_results:
        corridors = st.session_state.analysis_results['corridors']

        for corridor in corridors:
            bounds = corridor['bounds']
            x_coords = [bounds[0], bounds[2], bounds[2], bounds[0], bounds[0]]
            y_coords = [bounds[1], bounds[1], bounds[3], bounds[3], bounds[1]]

            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                fill="toself",
                mode='lines',
                name='Corridors' if corridor == corridors[0] else None,
                showlegend=(corridor == corridors[0]),
                fillcolor='rgba(150, 75, 0, 0.6)',  # Brown
                line=dict(color='brown', width=2),
                hovertemplate=f"<b>Corridor</b><br>Width: {corridor['width']:.1f} m<br>Length: {corridor['length']:.1f} m<extra></extra>"
            ))

    fig.update_layout(
        title="Enterprise Îlot Placement Results",
        xaxis_title="X Coordinate (m)",
        yaxis_title="Y Coordinate (m)",
        hovermode='closest',
        height=600,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )

    return fig

def show_statistics():
    """Display placement statistics"""
    st.subheader("📊 Placement Statistics")

    if not st.session_state.analysis_results:
        st.info("🎯 Run îlot placement analysis to see statistics")
        return

    results = st.session_state.analysis_results
    stats = results.get('statistics', {})

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Îlots", stats.get('total_ilots', 0))

    with col2:
        st.metric("Total Area Used", f"{stats.get('total_area_used', 0):.1f} m²")

    with col3:
        efficiency = stats.get('space_efficiency', 0) * 100
        st.metric("Space Efficiency", f"{efficiency:.1f}%")

    with col4:
        corridors = len(results.get('corridors', []))
        st.metric("Corridors Generated", corridors)

    # Size distribution comparison
    if 'distribution_achieved' in stats:
        st.subheader("📈 Size Distribution Analysis")

        achieved = stats['distribution_achieved']

        # Create comparison chart
        size_ranges = ['0-1', '1-3', '3-5', '5-10']
        target_values = []  # Would need to get from session state
        achieved_values = [achieved.get(sr, 0) for sr in size_ranges]

        fig = go.Figure(data=[
            go.Bar(name='Achieved', x=size_ranges, y=achieved_values, marker_color='lightblue')
        ])

        fig.update_layout(
            title='Îlot Size Distribution Achieved',
            xaxis_title='Size Range (m²)',
            yaxis_title='Percentage (%)',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

def show_detailed_results():
    """Show detailed analysis results"""
    st.subheader("🔍 Detailed Results")

    if not st.session_state.analysis_results:
        st.info("🎯 Run îlot placement analysis to see detailed results")
        return

    results = st.session_state.analysis_results

    # Validation results
    if 'validation' in results:
        validation = results['validation']

        if validation['is_valid']:
            st.success("✅ All placement constraints satisfied")
        else:
            st.warning(f"⚠️ {validation['total_violations']} constraint violations found")

            if validation['violations']:
                st.subheader("Constraint Violations:")
                for violation in validation['violations']:
                    st.error(f"- {violation['type']}: {violation}")

    # Îlot details table
    if 'placed_ilots' in results:
        st.subheader("📋 Placed Îlots")

        ilots_data = []
        for ilot in results['placed_ilots']:
            ilots_data.append({
                'ID': ilot['id'],
                'Size Range': ilot['size_range'],
                'Area (m²)': f"{ilot['area']:.2f}",
                'Position X': f"{ilot['position'][0]:.1f}",
                'Position Y': f"{ilot['position'][1]:.1f}",
                'Width': f"{ilot['width']:.2f}",
                'Height': f"{ilot['height']:.2f}"
            })

        df = pd.DataFrame(ilots_data)
        st.dataframe(df, use_container_width=True)

def show_export_options():
    """Show export and download options"""
    st.subheader("📄 Export Options")

    if not st.session_state.analysis_results:
        st.info("🎯 Run îlot placement analysis to enable export")
        return

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 Download Analysis Report (JSON)"):
            json_data = json.dumps(st.session_state.analysis_results, indent=2, default=str)
            st.download_button(
                label="💾 Download JSON Report",
                data=json_data,
                file_name="ilot_placement_report.json",
                mime="application/json"
            )

    with col2:
        if st.button("🖼️ Export Visualization (HTML)"):
            fig = create_ilot_visualization(True, True, True)
            html_content = fig.to_html()
            st.download_button(
                label="💾 Download HTML Visualization",
                data=html_content,
                file_name="ilot_placement_visualization.html",
                mime="text/html"
            )

if __name__ == "__main__":
    main()