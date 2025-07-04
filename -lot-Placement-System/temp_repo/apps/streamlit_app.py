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
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def create_basic_3d_plot(zones_data):
    """Create basic 3D plot when advanced visualization fails"""
    import plotly.graph_objects as go

    fig = go.Figure()

    wall_height = 3.0
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF']

    for i, zone in enumerate(zones_data):
        points = zone.get('points', [])
        if len(points) < 3:
            continue

        x_coords = [p[0] for p in points] + [points[0][0]]
        y_coords = [p[1] for p in points] + [points[0][1]]
        color = colors[i % len(colors)]

        # Floor
        fig.add_trace(go.Scatter3d(
            x=x_coords, y=y_coords, z=[0] * len(x_coords),
            mode='lines',
            line=dict(color=color, width=8),
            name=f'Floor - Zone {i+1}',
            showlegend=True
        ))

        # Ceiling
        fig.add_trace(go.Scatter3d(
            x=x_coords, y=y_coords, z=[wall_height] * len(x_coords),
            mode='lines',
            line=dict(color=color, width=6),
            name=f'Ceiling - Zone {i+1}',
            showlegend=False
        ))

        # Walls (vertical lines)
        for j in range(len(points)):
            fig.add_trace(go.Scatter3d(
                x=[points[j][0], points[j][0]],
                y=[points[j][1], points[j][1]], 
                z=[0, wall_height],
                mode='lines',
                line=dict(color=color, width=4),
                showlegend=False
            ))

    fig.update_layout(
        title="🌐 3D Building Model",
        scene=dict(
            xaxis_title="X (meters)",
            yaxis_title="Y (meters)",
            zaxis_title="Z (meters)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

def display_statistics(results):
    """Display detailed statistics with fixed Plotly configuration"""
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

def display_reports_section(analysis_results, zones_data):
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

def display_settings_section():
    """Display settings and configurations"""
    st.subheader("⚙️ Settings")
    st.write("Configure advanced settings here (under development)")

def show_analysis_interface():
    """Display analysis results and visualization"""
    zones_data = st.session_state.zones
    analysis_results = st.session_state.analysis_results

    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Analysis Dashboard", 
        "🎯 Interactive Plan", 
        "🌐 3D Visualization",
        "📈 Statistics", 
        "📄 Reports",
        "⚙️ Settings"
    ])

    with tab1:
        st.subheader("📊 Analysis Dashboard")
        if analysis_results:
            st.write("General analysis and key performance indicators.")
        else:
            st.info("Upload and analyze a DWG/DXF file to see the analysis dashboard")

    with tab2:
        st.subheader("🎯 Interactive Plan")
        if zones_data:
            show_visualization()
        else:
            st.info("Upload and analyze a DWG/DXF file to see the interactive plan")

    with tab3:
        st.subheader("🌐 3D Building Visualization")

        if zones_data:
            # 3D visualization controls
            col1, col2, col3 = st.columns(3)

            with col1:
                view_type = st.selectbox("3D View Type", [
                    "🏢 Building Model",
                    "🏗️ Construction View", 
                    "📐 Architectural Plan",
                    "🔧 Structural Frame"
                ])

            with col2:
                wall_height = st.slider("Wall Height (m)", 2.5, 5.0, 3.0, 0.1)

            with col3:
                show_furniture = st.checkbox("Show Furniture", True)

            # Create 3D visualization
            try:
                from src.advanced_visualization import AdvancedVisualizer
                visualizer = AdvancedVisualizer()

                if view_type == "🏢 Building Model":
                    fig_3d = visualizer.create_advanced_3d_model(zones_data, analysis_results, show_furniture, wall_height)
                elif view_type == "🏗️ Construction View":
                    fig_3d = visualizer.create_construction_plan_3d(zones_data, True)
                elif view_type == "📐 Architectural Plan":
                    fig_3d = visualizer.create_architectural_plan_3d(zones_data)
                else:  # Structural Frame
                    fig_3d = visualizer.create_structural_plan_3d(zones_data)

                st.plotly_chart(fig_3d, use_container_width=True, height=700)

                # 3D Model Information
                st.info(f"🌐 **3D Model Generated**: {len(zones_data)} zones rendered in 3D with {wall_height}m walls")

            except Exception as e:
                st.error(f"3D visualization error: {str(e)}")
                # Fallback basic 3D
                create_basic_3d_plot(zones_data)
        else:
            st.info("Upload and analyze a DWG/DXF file to see 3D visualization")

    with tab4:
        if analysis_results:
            display_statistics(analysis_results)
        else:
            st.info("Upload and analyze a DWG/DXF file to see statistics")

    with tab5:
        if analysis_results:
            display_reports_section(analysis_results, zones_data)
        else:
            st.info("Upload and analyze a DWG/DXF file to generate reports")

    with tab6:
        display_settings_section()

if __name__ == "__main__":
    main()