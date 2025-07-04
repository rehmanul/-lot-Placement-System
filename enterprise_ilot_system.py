"""
Enterprise Ilot Placement System
Professional CAD Analysis with Full Format Support
"""

import streamlit as st
import tempfile
import os
import json
from datetime import datetime
import traceback
from pathlib import Path

# Import enhanced modules
from src.dwg_parser import DWGParser
from src.enhanced_dwg_parser import EnhancedDWGParser
from src.pdf_parser import PDFParser
from src.placement_optimizer import PlacementOptimizer
from src.advanced_ai_models import AdvancedAIModels
from src.visualization import AdvancedVisualization
from src.cad_export import CADExporter
from src.export_utils import ExportManager

# Configure Streamlit
st.set_page_config(
    page_title="Enterprise Ilot System",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏗️ Enterprise Ilot Placement System</h1>
        <p>Professional CAD Analysis with Full Format Support</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'zones' not in st.session_state:
        st.session_state.zones = []
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}

    # Sidebar
    with st.sidebar:
        st.header("📂 File Upload")

        # File uploader with multiple formats
        uploaded_file = st.file_uploader(
            "Upload CAD/PDF File",
            type=['dwg', 'dxf', 'pdf'],
            help="Supports DWG, DXF, and PDF architectural plans"
        )

        if uploaded_file:
            st.success(f"✅ File loaded: {uploaded_file.name}")

            # File info
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.info(f"📊 File size: {file_size:.2f} MB")

            # Processing options
            st.header("⚙️ Processing Options")

            parsing_mode = st.selectbox(
                "Parsing Mode",
                ["Standard", "Enhanced", "Computer Vision"],
                help="Choose parsing strategy based on file complexity"
            )

            enable_ai = st.checkbox("🤖 Enable AI Analysis", value=True)
            enable_3d = st.checkbox("🎲 3D Visualization", value=False)

            # Process button
            if st.button("🚀 Process File", type="primary"):
                process_file(uploaded_file, parsing_mode, enable_ai, enable_3d)

    # Main content area
    if st.session_state.zones:
        display_analysis_results()
    else:
        display_welcome_screen()

def process_file(uploaded_file, parsing_mode, enable_ai, enable_3d):
    """Process uploaded file with comprehensive format support"""

    progress_container = st.container()

    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Determine file type
            filename = uploaded_file.name
            file_ext = os.path.splitext(filename.lower())[1]
            file_bytes = uploaded_file.getvalue()

            status_text.text("🔍 Analyzing file format...")
            progress_bar.progress(10)

            # Initialize appropriate parser
            zones = []

            if file_ext in ['.dwg', '.dxf']:
                status_text.text("📐 Processing CAD file...")
                progress_bar.progress(20)

                if parsing_mode == "Enhanced":
                    parser = EnhancedDWGParser()
                    result = parser.parse_advanced(file_bytes, filename)
                    if result['success']:
                        zones = result['zones']
                        st.session_state.metadata = result.get('metadata', {})
                    else:
                        st.error(f"❌ Enhanced parsing failed: {result['error']}")
                        return
                else:
                    parser = DWGParser()
                    zones = parser.parse_file_simple(file_bytes, filename)

            elif file_ext == '.pdf':
                status_text.text("📄 Processing PDF file...")
                progress_bar.progress(20)

                parser = PDFParser()
                zones = parser.parse_file_simple(file_bytes, filename)

            else:
                st.error(f"❌ Unsupported file format: {file_ext}")
                return

            progress_bar.progress(50)

            if not zones:
                st.error("❌ No valid zones found in the file")
                st.markdown("""
                <div class="info-box">
                    <strong>💡 Troubleshooting Tips:</strong><br>
                    • Ensure the file contains actual geometric data<br>
                    • Try different parsing modes<br>
                    • Check if the file is corrupted<br>
                    • For PDFs, ensure they contain vector graphics
                </div>
                """, unsafe_allow_html=True)
                return

            st.session_state.zones = zones

            # AI Analysis
            if enable_ai:
                status_text.text("🤖 Running AI analysis...")
                progress_bar.progress(70)

                try:
                    ai_analyzer = AdvancedAIModels()
                    analysis_results = ai_analyzer.analyze_comprehensive(zones)
                    st.session_state.analysis_results = analysis_results
                except Exception as e:
                    st.warning(f"⚠️ AI analysis failed: {str(e)}")
                    st.session_state.analysis_results = {}

            # Optimization
            status_text.text("🎯 Optimizing placements...")
            progress_bar.progress(90)

            try:
                optimizer = PlacementOptimizer()
                placement_results = optimizer.optimize_advanced(zones)
                st.session_state.analysis_results.update(placement_results)
            except Exception as e:
                st.warning(f"⚠️ Optimization failed: {str(e)}")

            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")

            # Success message
            st.markdown(f"""
            <div class="success-box">
                <strong>🎉 File processed successfully!</strong><br>
                📊 Found {len(zones)} zones<br>
                📁 Format: {file_ext.upper()}<br>
                ⚙️ Mode: {parsing_mode}
            </div>
            """, unsafe_allow_html=True)

            # Auto-refresh to show results
            st.rerun()

        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")

            # Detailed error information
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())

        finally:
            progress_bar.empty()
            status_text.empty()

def display_analysis_results():
    """Display comprehensive analysis results"""

    # Results tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🎯 Placements", 
        "📈 Analytics", 
        "📋 Export", 
        "🎨 Visualization"
    ])

    with tab1:
        display_overview_tab()

    with tab2:
        display_placements_tab()

    with tab3:
        display_analytics_tab()

    with tab4:
        display_export_tab()

    with tab5:
        display_visualization_tab()

def display_overview_tab():
    """Display overview of analysis results"""

    zones = st.session_state.zones
    results = st.session_state.analysis_results

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Zones", len(zones))

    with col2:
        total_area = sum(zone.get('area', 0) for zone in zones)
        st.metric("Total Area", f"{total_area:.1f} m²")

    with col3:
        total_placements = results.get('total_boxes', 0)
        st.metric("Placements", total_placements)

    with col4:
        efficiency = results.get('optimization', {}).get('total_efficiency', 0) * 100
        st.metric("Efficiency", f"{efficiency:.1f}%")

    # Zone details table
    st.subheader("📋 Zone Details")

    zone_data = []
    for i, zone in enumerate(zones):
        zone_data.append({
            'ID': zone.get('zone_id', f'Zone_{i}'),
            'Type': zone.get('zone_type', 'Unknown'),
            'Area (m²)': f"{zone.get('area', 0):.1f}",
            'Layer': zone.get('layer', 'Unknown'),
            'Entity': zone.get('entity_type', 'Unknown')
        })

    if zone_data:
        st.dataframe(zone_data, use_container_width=True)

def display_placements_tab():
    """Display placement optimization results"""

    results = st.session_state.analysis_results

    if 'placements' in results:
        st.subheader("🎯 Optimal Placements")

        # Placement summary
        placements = results['placements']

        if isinstance(placements, dict):
            for zone_name, zone_placements in placements.items():
                with st.expander(f"Zone: {zone_name} ({len(zone_placements)} placements)"):
                    placement_data = []

                    for i, placement in enumerate(zone_placements):
                        placement_data.append({
                            'Placement': i + 1,
                            'Position': f"({placement.get('position', [0, 0])[0]:.1f}, {placement.get('position', [0, 0])[1]:.1f})",
                            'Size': f"{placement.get('size', [0, 0])[0]:.1f} × {placement.get('size', [0, 0])[1]:.1f}",
                            'Area': f"{placement.get('area', 0):.1f} m²",
                            'Suitability': f"{placement.get('suitability_score', 0):.2f}"
                        })

                    if placement_data:
                        st.dataframe(placement_data, use_container_width=True)

        # Parameters used
        if 'parameters' in results:
            st.subheader("⚙️ Optimization Parameters")
            params = results['parameters']

            param_cols = st.columns(3)
            with param_cols[0]:
                st.metric("Box Size", f"{params.get('box_size', [0, 0])[0]} × {params.get('box_size', [0, 0])[1]} m")
            with param_cols[1]:
                st.metric("Margin", f"{params.get('margin', 0)} m")
            with param_cols[2]:
                st.metric("Rotation", "Enabled" if params.get('allow_rotation', False) else "Disabled")

    else:
        st.info("No placement optimization results available.")

def display_analytics_tab():
    """Display advanced analytics"""

    results = st.session_state.analysis_results

    if 'rooms' in results:
        st.subheader("🏠 Room Analysis")

        rooms = results['rooms']

        # Room type distribution
        room_types = {}
        confidences = []

        for room_info in rooms.values():
            room_type = room_info['type']
            room_types[room_type] = room_types.get(room_type, 0) + 1
            confidences.append(room_info['confidence'])

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Room Type Distribution:**")
            for room_type, count in room_types.items():
                st.write(f"• {room_type}: {count}")

        with col2:
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                st.metric("Average Confidence", f"{avg_confidence:.1%}")

    # Efficiency metrics
    if 'optimization' in results:
        st.subheader("📊 Efficiency Metrics")

        opt_results = results['optimization']

        efficiency_cols = st.columns(3)

        with efficiency_cols[0]:
            total_eff = opt_results.get('total_efficiency', 0)
            st.metric("Overall Efficiency", f"{total_eff:.1%}")

        with efficiency_cols[1]:
            space_util = opt_results.get('space_utilization', 0)
            st.metric("Space Utilization", f"{space_util:.1%}")

        with efficiency_cols[2]:
            placement_quality = opt_results.get('placement_quality', 0)
            st.metric("Placement Quality", f"{placement_quality:.1%}")

def display_export_tab():
    """Display export options"""

    st.subheader("📋 Export Results")

    zones = st.session_state.zones
    results = st.session_state.analysis_results

    if not zones:
        st.warning("No data to export.")
        return

    # Export formats
    export_cols = st.columns(3)

    with export_cols[0]:
        if st.button("📄 Export PDF Report", use_container_width=True):
            try:
                exporter = ExportManager()
                pdf_data = exporter.generate_pdf_report(zones, results)

                st.download_button(
                    "📥 Download PDF Report",
                    data=pdf_data,
                    file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )

                st.success("✅ PDF report generated!")

            except Exception as e:
                st.error(f"❌ PDF export failed: {str(e)}")

    with export_cols[1]:
        if st.button("📊 Export CSV Data", use_container_width=True):
            try:
                exporter = ExportManager()
                csv_data = exporter.export_to_csv(results)

                st.download_button(
                    "📥 Download CSV",
                    data=csv_data,
                    file_name=f"analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

                st.success("✅ CSV data exported!")

            except Exception as e:
                st.error(f"❌ CSV export failed: {str(e)}")

    with export_cols[2]:
        if st.button("📐 Export CAD Files", use_container_width=True):
            try:
                cad_exporter = CADExporter()

                # Create temporary files for CAD export
                with tempfile.TemporaryDirectory() as temp_dir:
                    dxf_path = os.path.join(temp_dir, "analysis_result.dxf")
                    cad_exporter.export_to_dxf(zones, results, dxf_path)

                    with open(dxf_path, 'rb') as f:
                        dxf_data = f.read()

                    st.download_button(
                        "📥 Download DXF",
                        data=dxf_data,
                        file_name=f"analysis_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dxf",
                        mime="application/octet-stream"
                    )

                st.success("✅ CAD file exported!")

            except Exception as e:
                st.error(f"❌ CAD export failed: {str(e)}")

    # JSON export
    st.subheader("🔧 Advanced Export")

    if st.button("📋 Export JSON Data"):
        try:
            exporter = ExportManager()
            json_data = exporter.export_to_json(results)

            st.download_button(
                "📥 Download JSON",
                data=json_data,
                file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

            st.success("✅ JSON data exported!")

        except Exception as e:
            st.error(f"❌ JSON export failed: {str(e)}")

def display_visualization_tab():
    """Display visualization options"""

    st.subheader("🎨 Visualization")

    zones = st.session_state.zones
    results = st.session_state.analysis_results

    if not zones:
        st.warning("No data to visualize.")
        return

    try:
        visualizer = AdvancedVisualization()

        # Create interactive plot
        fig = visualizer.create_interactive_plot(zones, results)

        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Unable to generate visualization.")

    except Exception as e:
        st.error(f"❌ Visualization failed: {str(e)}")

        # Fallback: simple zone list
        st.subheader("📋 Zone Summary")
        for i, zone in enumerate(zones):
            with st.expander(f"Zone {i+1}: {zone.get('zone_type', 'Unknown')}"):
                st.write(f"**Area:** {zone.get('area', 0):.1f} m²")
                st.write(f"**Layer:** {zone.get('layer', 'Unknown')}")
                st.write(f"**Type:** {zone.get('entity_type', 'Unknown')}")

                if 'points' in zone:
                    st.write(f"**Vertices:** {len(zone['points'])}")

def display_welcome_screen():
    """Display welcome screen when no file is loaded"""

    st.markdown("""
    <div class="info-box">
        <h3>🚀 Welcome to Enterprise Ilot System</h3>
        <p>Upload your CAD or PDF file to get started with professional architectural analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature highlights
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **📐 CAD Support**
        - DWG files (all versions)
        - DXF files with recovery
        - Multiple parsing strategies
        - Vector graphics extraction
        """)

    with col2:
        st.markdown("""
        **📄 PDF Analysis**
        - Vector graphics detection
        - Computer vision processing
        - Text-based extraction
        - Architectural plan recognition
        """)

    with col3:
        st.markdown("""
        **🤖 AI Features**
        - Intelligent room detection
        - Automated classification
        - Optimization algorithms
        - Quality analysis
        """)

    # Sample files
    st.subheader("📁 Sample Files")
    st.write("Try the system with these sample files:")

    sample_files = [
        "sample_files/Sample 1.dxf",
        "sample_files/anteen.dwg",
        "sample_files/apartment_plans.dwg"
    ]

    for sample_file in sample_files:
        if os.path.exists(sample_file):
            with open(sample_file, 'rb') as f:
                st.download_button(
                    f"📥 {os.path.basename(sample_file)}",
                    data=f.read(),
                    file_name=os.path.basename(sample_file),
                    mime="application/octet-stream"
                )

if __name__ == "__main__":
    main()