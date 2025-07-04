import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tempfile
import os
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import json
import math
import random
from shapely.geometry import Polygon, Point, LineString, box, MultiPolygon
from shapely.ops import unary_union
import ezdxf
from ezdxf import recover
import gc

# Page config
st.set_page_config(
    page_title="🏗️ Îlot Placement System", 
    page_icon="🏗️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

class IlotPlacementEngine:
    """Advanced îlot placement system for architectural plans"""

    def __init__(self):
        self.zones = []
        self.ilots = []
        self.corridors = []
        self.walls = []
        self.restricted_areas = []  # Blue areas (NO ENTREE)
        self.entrances = []  # Red areas (ENTREE/SORTIE)
        self.available_space = None
        self.plan_bounds = None

    def parse_dxf_file(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Parse DXF file and detect zones, walls, restricted areas, and entrances"""
        
        with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name

        try:
            try:
                doc = ezdxf.readfile(tmp_file_path)
                st.success("✅ DXF file loaded successfully")
            except ezdxf.DXFStructureError:
                doc, _ = recover.readfile(tmp_file_path)
                st.warning("⚠️ DXF file recovered with some structural issues")
            except Exception as e:
                st.error(f"❌ Cannot read DXF file: {str(e)}")
                return self._create_demo_data()

            modelspace = doc.modelspace()
            
            # Parse entities by color and layer
            walls = []
            restricted_areas = []
            entrances = []
            all_lines = []
            
            color_mapping = {
                1: 'red',      # Red - Entrances/Exits
                5: 'blue',     # Blue - Restricted areas
                7: 'black',    # Black/White - Walls
                256: 'bylayer' # By layer
            }

            for entity in modelspace:
                color_code = getattr(entity.dxf, 'color', 256)
                layer_name = getattr(entity.dxf, 'layer', '0').upper()
                
                if entity.dxftype() in ['LINE', 'LWPOLYLINE', 'POLYLINE']:
                    coords = self._extract_coordinates(entity)
                    if coords:
                        entity_info = {
                            'coords': coords,
                            'color': color_code,
                            'layer': layer_name,
                            'type': entity.dxftype()
                        }
                        
                        # Classify by color
                        if color_code == 1:  # Red - Entrances
                            entrances.append(entity_info)
                        elif color_code == 5:  # Blue - Restricted areas
                            restricted_areas.append(entity_info)
                        else:  # Black/default - Walls
                            walls.append(entity_info)
                            
                        all_lines.append(entity_info)

            # Calculate plan bounds
            if all_lines:
                all_coords = []
                for line in all_lines:
                    all_coords.extend(line['coords'])
                
                if all_coords:
                    xs = [p[0] for p in all_coords]
                    ys = [p[1] for p in all_coords]
                    self.plan_bounds = {
                        'min_x': min(xs), 'max_x': max(xs),
                        'min_y': min(ys), 'max_y': max(ys)
                    }

            self.walls = walls
            self.restricted_areas = restricted_areas
            self.entrances = entrances
            
            return {
                'walls': walls,
                'restricted_areas': restricted_areas,
                'entrances': entrances,
                'plan_bounds': self.plan_bounds,
                'entity_count': len(all_lines)
            }
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return self._create_demo_data()
        finally:
            try:
                os.unlink(tmp_file_path)
            except:
                pass

    def _extract_coordinates(self, entity) -> List[Tuple[float, float]]:
        """Extract coordinates from DXF entity"""
        coords = []
        
        try:
            if entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                coords = [(start.x, start.y), (end.x, end.y)]
                
            elif entity.dxftype() == 'LWPOLYLINE':
                coords = [(point[0], point[1]) for point in entity.get_points()]
                
            elif entity.dxftype() == 'POLYLINE':
                coords = [(vertex.dxf.location.x, vertex.dxf.location.y) 
                         for vertex in entity.vertices]
        except:
            pass
            
        return coords

    def _create_demo_data(self) -> Dict[str, Any]:
        """Create demo data for testing when file loading fails"""
        
        # Create a simple rectangular plan
        walls = [
            {'coords': [(0, 0), (50, 0)], 'color': 7, 'layer': 'WALLS', 'type': 'LINE'},
            {'coords': [(50, 0), (50, 30)], 'color': 7, 'layer': 'WALLS', 'type': 'LINE'},
            {'coords': [(50, 30), (0, 30)], 'color': 7, 'layer': 'WALLS', 'type': 'LINE'},
            {'coords': [(0, 30), (0, 0)], 'color': 7, 'layer': 'WALLS', 'type': 'LINE'},
        ]
        
        # Add some internal walls
        walls.extend([
            {'coords': [(20, 0), (20, 15)], 'color': 7, 'layer': 'WALLS', 'type': 'LINE'},
            {'coords': [(0, 15), (20, 15)], 'color': 7, 'layer': 'WALLS', 'type': 'LINE'},
        ])
        
        # Restricted areas (blue)
        restricted_areas = [
            {'coords': [(5, 25), (10, 25), (10, 28), (5, 28), (5, 25)], 'color': 5, 'layer': 'RESTRICTED', 'type': 'LWPOLYLINE'},
            {'coords': [(40, 5), (45, 5), (45, 10), (40, 10), (40, 5)], 'color': 5, 'layer': 'RESTRICTED', 'type': 'LWPOLYLINE'},
        ]
        
        # Entrances (red)
        entrances = [
            {'coords': [(24, 0), (26, 0)], 'color': 1, 'layer': 'ENTRANCES', 'type': 'LINE'},
            {'coords': [(0, 8), (0, 10)], 'color': 1, 'layer': 'ENTRANCES', 'type': 'LINE'},
        ]
        
        self.plan_bounds = {'min_x': 0, 'max_x': 50, 'min_y': 0, 'max_y': 30}
        self.walls = walls
        self.restricted_areas = restricted_areas
        self.entrances = entrances
        
        return {
            'walls': walls,
            'restricted_areas': restricted_areas,
            'entrances': entrances,
            'plan_bounds': self.plan_bounds,
            'entity_count': len(walls) + len(restricted_areas) + len(entrances)
        }

    def calculate_available_zones(self) -> List[Polygon]:
        """Calculate available zones for îlot placement"""
        if not self.plan_bounds:
            return []
        
        # Create overall boundary
        boundary = box(
            self.plan_bounds['min_x'], 
            self.plan_bounds['min_y'],
            self.plan_bounds['max_x'], 
            self.plan_bounds['max_y']
        )
        
        # Create polygons for restricted areas and entrances
        restricted_polygons = []
        
        # Process restricted areas (blue zones)
        for area in self.restricted_areas:
            if len(area['coords']) >= 3:
                try:
                    poly = Polygon(area['coords'])
                    if poly.is_valid:
                        # Buffer to create exclusion zone
                        restricted_polygons.append(poly.buffer(0.5))
                except:
                    pass
        
        # Process entrances (red zones) - create larger exclusion zones
        for entrance in self.entrances:
            if len(entrance['coords']) >= 2:
                try:
                    # Create buffer around entrance line
                    line = LineString(entrance['coords'])
                    # Large buffer to avoid placing îlots near entrances
                    restricted_polygons.append(line.buffer(2.0))
                except:
                    pass
        
        # Subtract restricted areas from boundary
        if restricted_polygons:
            restricted_union = unary_union(restricted_polygons)
            available_space = boundary.difference(restricted_union)
        else:
            available_space = boundary
        
        # Convert to list of polygons
        if isinstance(available_space, MultiPolygon):
            zones = [poly for poly in available_space.geoms if poly.area > 1.0]
        elif isinstance(available_space, Polygon) and available_space.area > 1.0:
            zones = [available_space]
        else:
            zones = []
        
        self.zones = zones
        return zones

    def generate_ilot_layout(self, profile: Dict[str, float], total_area: float) -> List[Dict]:
        """Generate îlot layout based on profile and constraints"""
        
        # Calculate number of îlots for each size category
        ilot_sizes = {
            'small': (0.5, 1.0),    # 0-1 m²
            'medium_small': (1.0, 3.0),  # 1-3 m²
            'medium': (3.0, 5.0),   # 3-5 m²
            'large': (5.0, 10.0)    # 5-10 m²
        }
        
        # Calculate target number of îlots
        size_categories = ['small', 'medium_small', 'medium', 'large']
        profile_percentages = [
            profile.get('small', 10),
            profile.get('medium_small', 25),
            profile.get('medium', 30),
            profile.get('large', 35)
        ]
        
        # Estimate total number of îlots based on available area
        estimated_total_ilots = int(total_area / 4.0)  # Average 4m² per îlot
        
        ilots = []
        ilot_id = 1
        
        for category, percentage in zip(size_categories, profile_percentages):
            count = int((percentage / 100.0) * estimated_total_ilots)
            min_size, max_size = ilot_sizes[category]
            
            for _ in range(count):
                # Random size within category
                area = random.uniform(min_size, max_size)
                
                # Try to place îlot
                placement = self._place_single_ilot(area, ilot_id)
                if placement:
                    ilots.append(placement)
                    ilot_id += 1
        
        self.ilots = ilots
        return ilots

    def _place_single_ilot(self, target_area: float, ilot_id: int) -> Optional[Dict]:
        """Place a single îlot in available space"""
        
        if not self.zones:
            return None
        
        # Try to place in each available zone
        for zone in self.zones:
            if zone.area < target_area:
                continue
            
            # Get zone bounds
            minx, miny, maxx, maxy = zone.bounds
            
            # Calculate îlot dimensions (try to make roughly square)
            side_length = math.sqrt(target_area)
            
            # Try multiple random positions
            for attempt in range(50):
                # Random position within zone bounds
                x = random.uniform(minx, maxx - side_length)
                y = random.uniform(miny, maxy - side_length)
                
                # Create îlot rectangle
                ilot_rect = box(x, y, x + side_length, y + side_length)
                
                # Check if îlot fits within zone and doesn't overlap existing îlots
                if zone.contains(ilot_rect) and not self._overlaps_existing_ilots(ilot_rect):
                    return {
                        'id': ilot_id,
                        'geometry': ilot_rect,
                        'area': target_area,
                        'center': (x + side_length/2, y + side_length/2),
                        'bounds': (x, y, x + side_length, y + side_length)
                    }
        
        return None

    def _overlaps_existing_ilots(self, new_ilot: Polygon) -> bool:
        """Check if new îlot overlaps with existing ones"""
        buffer_distance = 0.5  # Minimum distance between îlots
        
        for existing in self.ilots:
            if new_ilot.distance(existing['geometry']) < buffer_distance:
                return True
        return False

    def generate_corridors(self, corridor_width: float = 1.5) -> List[Dict]:
        """Generate corridors between facing îlot rows"""
        corridors = []
        
        if len(self.ilots) < 2:
            return corridors
        
        # Group îlots by approximate rows (same Y coordinate range)
        rows = self._group_ilots_into_rows()
        
        corridor_id = 1
        for i, row1 in enumerate(rows):
            for j, row2 in enumerate(rows[i+1:], i+1):
                # Check if rows face each other
                corridor = self._create_corridor_between_rows(row1, row2, corridor_width, corridor_id)
                if corridor:
                    corridors.append(corridor)
                    corridor_id += 1
        
        self.corridors = corridors
        return corridors

    def _group_ilots_into_rows(self, tolerance: float = 2.0) -> List[List[Dict]]:
        """Group îlots into rows based on Y coordinates"""
        if not self.ilots:
            return []
        
        # Sort îlots by Y coordinate
        sorted_ilots = sorted(self.ilots, key=lambda x: x['center'][1])
        
        rows = []
        current_row = [sorted_ilots[0]]
        current_y = sorted_ilots[0]['center'][1]
        
        for ilot in sorted_ilots[1:]:
            if abs(ilot['center'][1] - current_y) <= tolerance:
                current_row.append(ilot)
            else:
                if len(current_row) >= 2:  # Only keep rows with multiple îlots
                    rows.append(current_row)
                current_row = [ilot]
                current_y = ilot['center'][1]
        
        if len(current_row) >= 2:
            rows.append(current_row)
        
        return rows

    def _create_corridor_between_rows(self, row1: List[Dict], row2: List[Dict], 
                                    width: float, corridor_id: int) -> Optional[Dict]:
        """Create corridor between two rows of îlots"""
        
        # Calculate average Y positions
        avg_y1 = sum(ilot['center'][1] for ilot in row1) / len(row1)
        avg_y2 = sum(ilot['center'][1] for ilot in row2) / len(row2)
        
        # Check if rows are reasonably close and parallel
        distance = abs(avg_y2 - avg_y1)
        if distance > 10:  # Too far apart
            return None
        
        # Find overlapping X range
        min_x1 = min(ilot['bounds'][0] for ilot in row1)
        max_x1 = max(ilot['bounds'][2] for ilot in row1)
        min_x2 = min(ilot['bounds'][0] for ilot in row2)
        max_x2 = max(ilot['bounds'][2] for ilot in row2)
        
        overlap_start = max(min_x1, min_x2)
        overlap_end = min(max_x1, max_x2)
        
        if overlap_start >= overlap_end:
            return None
        
        # Create corridor in the middle
        corridor_y = (avg_y1 + avg_y2) / 2
        corridor_rect = box(
            overlap_start, 
            corridor_y - width/2,
            overlap_end, 
            corridor_y + width/2
        )
        
        return {
            'id': corridor_id,
            'geometry': corridor_rect,
            'width': width,
            'length': overlap_end - overlap_start,
            'bounds': (overlap_start, corridor_y - width/2, overlap_end, corridor_y + width/2)
        }

    def visualize_plan(self) -> go.Figure:
        """Create interactive visualization of the plan with îlots and corridors"""
        
        fig = go.Figure()
        
        # Add walls (black lines)
        for wall in self.walls:
            if len(wall['coords']) >= 2:
                xs = [coord[0] for coord in wall['coords']]
                ys = [coord[1] for coord in wall['coords']]
                
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode='lines',
                    line=dict(color='black', width=3),
                    name='Murs (Walls)',
                    showlegend=len(fig.data) == 0 or not any('Murs' in trace.name for trace in fig.data)
                ))
        
        # Add restricted areas (blue)
        for area in self.restricted_areas:
            if len(area['coords']) >= 3:
                xs = [coord[0] for coord in area['coords']] + [area['coords'][0][0]]
                ys = [coord[1] for coord in area['coords']] + [area['coords'][0][1]]
                
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    fill='toself',
                    fillcolor='rgba(0, 150, 255, 0.3)',
                    line=dict(color='blue', width=2),
                    name='No Entrée (Restricted)',
                    showlegend=len([t for t in fig.data if 'No Entrée' in str(t.name)]) == 0
                ))
        
        # Add entrances (red)
        for entrance in self.entrances:
            xs = [coord[0] for coord in entrance['coords']]
            ys = [coord[1] for coord in entrance['coords']]
            
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode='lines',
                line=dict(color='red', width=5),
                name='Entrée/Sortie',
                showlegend=len([t for t in fig.data if 'Entrée' in str(t.name)]) == 0
            ))
        
        # Add îlots (pink/red rectangles)
        for ilot in self.ilots:
            bounds = ilot['bounds']
            xs = [bounds[0], bounds[2], bounds[2], bounds[0], bounds[0]]
            ys = [bounds[1], bounds[1], bounds[3], bounds[3], bounds[1]]
            
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                fill='toself',
                fillcolor='rgba(255, 100, 150, 0.6)',
                line=dict(color='darkred', width=1),
                name=f'Îlot {ilot["id"]} ({ilot["area"]:.1f}m²)',
                text=f'{ilot["area"]:.1f}m²',
                textposition='middle center'
            ))
        
        # Add corridors (light gray)
        for corridor in self.corridors:
            bounds = corridor['bounds']
            xs = [bounds[0], bounds[2], bounds[2], bounds[0], bounds[0]]
            ys = [bounds[1], bounds[1], bounds[3], bounds[3], bounds[1]]
            
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                fill='toself',
                fillcolor='rgba(200, 200, 200, 0.5)',
                line=dict(color='gray', width=1),
                name=f'Corridor {corridor["id"]}',
                showlegend=len([t for t in fig.data if 'Corridor' in str(t.name)]) == 0
            ))
        
        # Update layout
        fig.update_layout(
            title="🏗️ Plan avec Îlots et Corridors",
            xaxis_title="X (mètres)",
            yaxis_title="Y (mètres)",
            showlegend=True,
            hovermode='closest',
            template='plotly_white',
            height=600
        )
        
        # Set equal aspect ratio
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
        
        return fig

    def export_results(self) -> Dict[str, Any]:
        """Export results for download"""
        
        summary = {
            'total_ilots': len(self.ilots),
            'total_corridors': len(self.corridors),
            'total_ilot_area': sum(ilot['area'] for ilot in self.ilots),
            'ilot_details': [
                {
                    'id': ilot['id'],
                    'area_m2': round(ilot['area'], 2),
                    'center_x': round(ilot['center'][0], 2),
                    'center_y': round(ilot['center'][1], 2)
                }
                for ilot in self.ilots
            ],
            'corridor_details': [
                {
                    'id': corridor['id'],
                    'width': corridor['width'],
                    'length': round(corridor['length'], 2)
                }
                for corridor in self.corridors
            ]
        }
        
        return summary


# Streamlit UI
def main():
    st.title("🏗️ Système de Placement d'Îlots Professionnel")
    st.markdown("---")
    
    # Initialize session state
    if 'engine' not in st.session_state:
        st.session_state.engine = IlotPlacementEngine()
    
    engine = st.session_state.engine
    
    # Sidebar for controls
    with st.sidebar:
        st.header("📋 Configuration")
        
        # File upload
        st.subheader("1. Charger le Plan")
        uploaded_file = st.file_uploader(
            "Choisir fichier DXF/DWG",
            type=['dxf', 'dwg'],
            help="Télécharger un plan architectural en format DXF ou DWG"
        )
        
        # Îlot profile configuration
        st.subheader("2. Profil des Îlots")
        st.write("Définir les proportions par taille:")
        
        small_pct = st.slider("0-1 m²", 0, 50, 10, 5, key="small")
        medium_small_pct = st.slider("1-3 m²", 0, 50, 25, 5, key="medium_small")
        medium_pct = st.slider("3-5 m²", 0, 50, 30, 5, key="medium")
        large_pct = st.slider("5-10 m²", 0, 50, 35, 5, key="large")
        
        total_pct = small_pct + medium_small_pct + medium_pct + large_pct
        if total_pct != 100:
            st.warning(f"⚠️ Total: {total_pct}% (devrait être 100%)")
        
        profile = {
            'small': small_pct,
            'medium_small': medium_small_pct,
            'medium': medium_pct,
            'large': large_pct
        }
        
        # Corridor configuration
        st.subheader("3. Configuration Corridors")
        corridor_width = st.slider("Largeur corridor (m)", 0.5, 3.0, 1.5, 0.1)
        
        # Action buttons
        st.subheader("4. Actions")
        if st.button("🔍 Analyser le Plan", type="primary"):
            if uploaded_file:
                with st.spinner("Analyse du fichier..."):
                    file_bytes = uploaded_file.read()
                    result = engine.parse_dxf_file(file_bytes, uploaded_file.name)
                    st.session_state.plan_loaded = True
                    st.success("✅ Plan analysé avec succès!")
            else:
                with st.spinner("Chargement des données de démonstration..."):
                    result = engine._create_demo_data()
                    st.session_state.plan_loaded = True
                    st.info("📊 Utilisation des données de démonstration")
        
        if st.button("🎯 Placer les Îlots"):
            if hasattr(st.session_state, 'plan_loaded'):
                with st.spinner("Placement des îlots..."):
                    # Calculate available zones
                    zones = engine.calculate_available_zones()
                    total_area = sum(zone.area for zone in zones)
                    
                    # Generate îlot layout
                    ilots = engine.generate_ilot_layout(profile, total_area)
                    
                    # Generate corridors
                    corridors = engine.generate_corridors(corridor_width)
                    
                    st.session_state.ilots_placed = True
                    st.success(f"✅ {len(ilots)} îlots placés avec {len(corridors)} corridors!")
            else:
                st.error("❌ Veuillez d'abord analyser un plan")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📐 Visualisation du Plan")
        
        if hasattr(st.session_state, 'plan_loaded'):
            if hasattr(st.session_state, 'ilots_placed'):
                # Show complete visualization
                fig = engine.visualize_plan()
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Show plan analysis only
                st.info("Plan chargé. Cliquez sur 'Placer les Îlots' pour continuer.")
                fig = engine.visualize_plan()
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📁 Téléchargez un fichier DXF/DWG ou utilisez les données de démonstration")
    
    with col2:
        st.subheader("📊 Statistiques")
        
        if hasattr(st.session_state, 'ilots_placed') and engine.ilots:
            # Statistics
            total_ilots = len(engine.ilots)
            total_area = sum(ilot['area'] for ilot in engine.ilots)
            total_corridors = len(engine.corridors)
            
            st.metric("Nombre d'îlots", total_ilots)
            st.metric("Surface totale îlots", f"{total_area:.1f} m²")
            st.metric("Nombre de corridors", total_corridors)
            
            # Size distribution
            st.subheader("🔢 Répartition par Taille")
            
            size_counts = {'0-1m²': 0, '1-3m²': 0, '3-5m²': 0, '5-10m²': 0}
            for ilot in engine.ilots:
                area = ilot['area']
                if area <= 1:
                    size_counts['0-1m²'] += 1
                elif area <= 3:
                    size_counts['1-3m²'] += 1
                elif area <= 5:
                    size_counts['3-5m²'] += 1
                else:
                    size_counts['5-10m²'] += 1
            
            for size_range, count in size_counts.items():
                percentage = (count / total_ilots * 100) if total_ilots > 0 else 0
                st.write(f"{size_range}: {count} ({percentage:.1f}%)")
            
            # Export results
            st.subheader("💾 Export")
            if st.button("📥 Télécharger Résultats"):
                results = engine.export_results()
                st.download_button(
                    label="📄 Télécharger JSON",
                    data=json.dumps(results, indent=2),
                    file_name="ilot_placement_results.json",
                    mime="application/json"
                )
        else:
            st.info("Aucune donnée disponible.\nVeuillez analyser un plan et placer les îlots.")
    
    # Legend
    with st.expander("🗺️ Légende"):
        st.write("""
        **Couleurs du plan:**
        - 🔴 **Rouge**: Entrées/Sorties (ENTREE/SORTIE) - Zones à éviter
        - 🔵 **Bleu**: Zones restreintes (NO ENTREE) - Escaliers, ascenseurs
        - ⚫ **Noir**: Murs (MUR) - Structure du bâtiment
        - 🟣 **Rose**: Îlots placés avec surface indiquée
        - ⬜ **Gris**: Corridors entre rangées d'îlots
        """)


if __name__ == "__main__":
    main()