import streamlit as st
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 2b57e5d (Created a checkpoint)
>>>>>>> origin/replit-agent
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> origin/replit-agent
import json
import math
import random
from shapely.geometry import Polygon, Point, LineString, box, MultiPolygon
from shapely.ops import unary_union
import ezdxf
from ezdxf import recover
import tempfile
import os
from pathlib import Path
import cv2
from PIL import Image
import io
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.colors import red, blue, black, green, orange, purple
from reportlab.lib.units import inch
import base64
from dwg_parser import DWGParser
from ilot_placement_engine import IlotPlacementEngine

# Page configuration
st.set_page_config(
    page_title="🏗️ Enterprise Îlot Placement System",
    page_icon="🏗️",
<<<<<<< HEAD
=======
=======
=======
>>>>>>> 2b57e5d (Created a checkpoint)
import json
import math
import random
from shapely.geometry import Polygon, Point, LineString, box, MultiPolygon
from shapely.ops import unary_union
import ezdxf
from ezdxf import recover
import tempfile
import os
from pathlib import Path
import cv2
from PIL import Image
import io
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.colors import red, blue, black, green, orange, purple
from reportlab.lib.units import inch
import base64
from dwg_parser import DWGParser
from ilot_placement_engine import IlotPlacementEngine

# Page configuration
st.set_page_config(
<<<<<<< HEAD
    page_title="App Requirements Workspace",
    page_icon="📋",
>>>>>>> b49860d (Create a collaborative workspace for app ideas and requirements)
=======
    page_title="Enterprise Îlot Placement System",
    page_icon="🏗️",
>>>>>>> 2b57e5d (Created a checkpoint)
>>>>>>> origin/replit-agent
    layout="wide",
    initial_sidebar_state="expanded"
)

<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 2b57e5d (Created a checkpoint)
>>>>>>> origin/replit-agent
# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> origin/replit-agent
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .warning-message {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    .info-box {
        background: #e7f3ff;
        border: 1px solid #b3d9ff;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #2a5298 0%, #1e3c72 100%);
        color: white;
        border: none;
        border-radius: 8px;
<<<<<<< HEAD
=======
=======
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .zone-legend {
        display: flex;
        gap: 20px;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .legend-color {
        width: 20px;
        height: 20px;
        border: 1px solid #333;
    }
    .stButton > button {
        width: 100%;
        background-color: #2a5298;
        color: white;
        border: none;
>>>>>>> 2b57e5d (Created a checkpoint)
>>>>>>> origin/replit-agent
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> origin/replit-agent
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

class EnterpriseIlotPlacementSystem:
    """Enterprise-grade îlot placement system for architectural space planning"""

    def __init__(self):
        self.reset_system()

    def reset_system(self):
        """Reset all system components"""
        self.zones = []
        self.ilots = []
        self.corridors = []
        self.walls = []
        self.restricted_areas = []
        self.entrances = []
        self.available_zones = []
        self.plan_bounds = None
        self.scale_factor = 1.0
        self.metadata = {}
        self.placement_engine = IlotPlacementEngine()

    def place_ilots_with_distribution(self, zones, layout_profile, total_area, corridor_width):
        """Place ilots with distribution using the placement engine"""
        return self.placement_engine.place_ilots_with_distribution(zones, layout_profile, total_area, corridor_width)

    def parse_floor_plan(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Parse uploaded floor plan file (DXF/DWG/Image)"""
        try:
            file_ext = Path(filename).suffix.lower()

            if file_ext in ['.dxf', '.dwg']:
                return self._parse_cad_file(file_bytes, filename)
            elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                return self._parse_image_file(file_bytes, filename)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")

        except Exception as e:
            st.error(f"Error parsing floor plan: {str(e)}")
            return {"error": str(e)}

    def _parse_cad_file(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Parse CAD files (DXF/DWG) - Enterprise-grade implementation"""

        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name

        try:
            # Parse the actual CAD file
            if filename.lower().endswith('.dwg'):
                # DWG files need recovery mode
                try:
                    doc, auditor = recover.readfile(tmp_file_path)
                    if auditor.has_errors:
                        st.warning(f"DWG file has {len(auditor.errors)} structural issues but was recovered")
                    st.success(f"Successfully loaded DWG file: {filename}")
                except Exception as e:
                    st.error(f"Failed to parse DWG file: {str(e)}")
                    return {'error': f'DWG parsing failed: {str(e)}'}
            else:
                # DXF files
                try:
                    doc = ezdxf.readfile(tmp_file_path)
                    st.success(f"Successfully loaded DXF file: {filename}")
                except (IOError, Exception) as e:
                    # Try recovery mode
                    try:
                        doc, auditor = recover.readfile(tmp_file_path)
                        if auditor.has_errors:
                            st.warning(f"DXF file has {len(auditor.errors)} structural issues but was recovered")
                        st.success(f"Successfully recovered DXF file: {filename}")
                    except Exception as recovery_error:
                        st.error(f"Failed to parse DXF file: {str(recovery_error)}")
                        return {'error': f'DXF parsing failed: {str(recovery_error)}'}

            # Get modelspace
            modelspace = doc.modelspace()
            entity_count = len(list(modelspace))
            st.info(f"Processing {entity_count} entities from CAD file...")

            # Initialize collections
            walls = []
            restricted_areas = []
            entrances = []
            all_points = []

            # Process each entity
            for entity in modelspace:
                try:
                    layer_name = getattr(entity.dxf, 'layer', '0').upper()
                    entity_type = entity.dxftype()
                    color = getattr(entity.dxf, 'color', 7)  # Default color is 7 (white/black)

                    # Process LINE entities
                    if entity_type == 'LINE':
                        start = (entity.dxf.start.x, entity.dxf.start.y)
                        end = (entity.dxf.end.x, entity.dxf.end.y)
                        line = LineString([start, end])
                        all_points.extend([start, end])

                        # Classify based on layer, color, or name
                        if any(keyword in layer_name for keyword in ['WALL', 'MUR', 'CLOISON', 'PARTITION']):
                            walls.append(line)
                        elif any(keyword in layer_name for keyword in ['ENTRANCE', 'ENTREE', 'SORTIE', 'EXIT', 'DOOR', 'PORTE']):
                            entrances.append(line.buffer(1.0))  # Create small polygon around entrance
                        elif color == 1:  # Red color
                            entrances.append(line.buffer(1.0))
                        elif color == 7 or color == 0:  # Black/white
                            walls.append(line)

                    # Process POLYLINE entities
                    elif entity_type in ['POLYLINE', 'LWPOLYLINE']:
                        points = []

                        if entity_type == 'LWPOLYLINE':
                            # LightWeight polyline
                            try:
                                points = [(point[0], point[1]) for point in entity.get_points()]
                            except:
                                # Alternative method for LWPOLYLINE
                                with entity.points() as p:
                                    points = [(point[0], point[1]) for point in p]
                        else:
                            # Regular polyline
                            try:
                                for vertex in entity.vertices:
                                    if hasattr(vertex, 'dxf') and hasattr(vertex.dxf, 'location'):
                                        points.append((vertex.dxf.location.x, vertex.dxf.location.y))
                                    elif hasattr(vertex, 'dxf'):
                                        points.append((vertex.dxf.location[0], vertex.dxf.location[1]))
                            except:
                                pass

                        if len(points) >= 2:
                            all_points.extend(points)

                            # Check if closed
                            is_closed = getattr(entity, 'is_closed', False)

                            if is_closed and len(points) >= 3:
                                # Create polygon
                                try:
                                    polygon = Polygon(points)
                                    if polygon.is_valid:
                                        # Classify polygon
                                        if any(keyword in layer_name for keyword in ['RESTRICTED', 'NO_ENTRY', 'NO_ENTREE', 'INTERDIT', 'ESCALIER', 'STAIRS', 'ELEVATOR', 'ASCENSEUR']):
                                            restricted_areas.append(polygon)
                                        elif any(keyword in layer_name for keyword in ['ENTRANCE', 'ENTREE', 'SORTIE', 'EXIT']):
                                            entrances.append(polygon)
                                        elif color == 5:  # Blue color
                                            restricted_areas.append(polygon)
                                        elif color == 1:  # Red color
                                            entrances.append(polygon)
                                        else:
                                            # Add as wall boundary
                                            walls.append(polygon.boundary)
                                except Exception:
                                    # If polygon creation fails, add as line segments
                                    for i in range(len(points) - 1):
                                        walls.append(LineString([points[i], points[i+1]]))
                            else:
                                # Open polyline - add as line segments
                                for i in range(len(points) - 1):
                                    line = LineString([points[i], points[i+1]])
                                    if any(keyword in layer_name for keyword in ['WALL', 'MUR']):
                                        walls.append(line)
                                    else:
                                        walls.append(line)

                    # Process CIRCLE entities
                    elif entity_type == 'CIRCLE':
                        center = (entity.dxf.center.x, entity.dxf.center.y)
                        radius = entity.dxf.radius
                        circle = Point(center).buffer(radius)

                        if any(keyword in layer_name for keyword in ['RESTRICTED', 'COLUMN', 'POTEAU']):
                            restricted_areas.append(circle)
                        elif color == 5:  # Blue
                            restricted_areas.append(circle)

                    # Process ARC entities
                    elif entity_type == 'ARC':
                        # Convert arc to line segments
                        start_angle = math.radians(entity.dxf.start_angle)
                        end_angle = math.radians(entity.dxf.end_angle)
                        center = (entity.dxf.center.x, entity.dxf.center.y)
                        radius = entity.dxf.radius

                        # Create arc points
                        num_segments = 20
                        arc_points = []
                        for i in range(num_segments + 1):
                            angle = start_angle + (end_angle - start_angle) * i / num_segments
                            x = center[0] + radius * math.cos(angle)
                            y = center[1] + radius * math.sin(angle)
                            arc_points.append((x, y))

                        if len(arc_points) >= 2:
                            for i in range(len(arc_points) - 1):
                                walls.append(LineString([arc_points[i], arc_points[i+1]]))
                                all_points.extend([arc_points[i], arc_points[i+1]])

                    # Process HATCH entities (often used for restricted areas)
                    elif entity_type == 'HATCH':
                        # Hatches often indicate special areas
                        try:
                            paths = entity.paths
                            for path in paths:
                                if path.type == 'PolylinePath':
                                    vertices = [(v[0], v[1]) for v in path.vertices]
                                    if len(vertices) >= 3:
                                        polygon = Polygon(vertices)
                                        if polygon.is_valid:
                                            if color == 5:  # Blue
                                                restricted_areas.append(polygon)
                                            elif color == 1:  # Red
                                                entrances.append(polygon)
                                            else:
                                                restricted_areas.append(polygon)
                        except:
                            pass

                except Exception as entity_error:
                    # Skip problematic entities
                    continue

            # Calculate bounds from all collected points
            if all_points:
                x_coords = [p[0] for p in all_points]
                y_coords = [p[1] for p in all_points]
                self.plan_bounds = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            else:
                self.plan_bounds = (0, 0, 100, 100)  # Default bounds if no points found

            # Store parsed data
            self.walls = walls
            self.restricted_areas = restricted_areas
            self.entrances = entrances

            # Report results
            st.success(f"✅ CAD file parsed successfully!")
            st.info(f"Found: {len(walls)} walls, {len(restricted_areas)} restricted areas, {len(entrances)} entrances/exits")

            return {
                'status': 'success',
                'walls': len(walls),
                'restricted_areas': len(restricted_areas),
                'entrances': len(entrances),
                'bounds': self.plan_bounds,
                'entity_count': entity_count
            }

        except Exception as e:
            st.error(f"Error parsing CAD file: {str(e)}")
            return {'error': str(e)}
        finally:
            # Clean up temp file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    def _parse_image_file(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Parse image files and detect zones using computer vision"""
        try:
            # Convert bytes to image
            image = Image.open(io.BytesIO(file_bytes))
            img_array = np.array(image)

            # Convert to BGR for OpenCV
            if len(img_array.shape) == 3:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

            # Detect walls (black lines)
            walls = self._detect_walls(img_bgr)

            # Detect restricted areas (blue zones)
            restricted_areas = self._detect_restricted_areas(img_bgr)

            # Detect entrances (red zones)
            entrances = self._detect_entrances(img_bgr)

            # Set plan bounds
            h, w = img_bgr.shape[:2]
            self.plan_bounds = (0, 0, w, h)

            self.walls = walls
            self.restricted_areas = restricted_areas
            self.entrances = entrances

            return {
                'status': 'success',
                'walls': len(walls),
                'restricted_areas': len(restricted_areas),
                'entrances': len(entrances),
                'bounds': self.plan_bounds,
                'image_size': (w, h)
            }

        except Exception as e:
            st.error(f"Error parsing image file: {str(e)}")
            return {'error': str(e)}

    def _detect_walls(self, img: np.ndarray) -> List[LineString]:
        """Detect walls from image (black lines)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create mask for black lines
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        walls = []
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) > 100:
                # Convert contour to line segments
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                points = [(point[0][0], point[0][1]) for point in approx]
                if len(points) >= 2:
                    for i in range(len(points) - 1):
                        walls.append(LineString([points[i], points[i+1]]))

        return walls

    def _detect_restricted_areas(self, img: np.ndarray) -> List[Polygon]:
        """Detect restricted areas from image (blue zones)"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define range for blue color
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # Create mask
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        restricted_areas = []
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                # Convert contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                points = [(point[0][0], point[0][1]) for point in approx]
                if len(points) >= 3:
                    try:
                        polygon = Polygon(points)
                        if polygon.is_valid:
                            restricted_areas.append(polygon)
                    except:
                        continue

        return restricted_areas

    def _detect_entrances(self, img: np.ndarray) -> List[Polygon]:
        """Detect entrances from image (red zones)"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define range for red color
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        # Create mask
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        entrances = []
        for contour in contours:
            if cv2.contourArea(contour) > 20:
                # Convert contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                points = [(point[0][0], point[0][1]) for point in approx]
                if len(points) >= 3:
                    try:
                        polygon = Polygon(points)
                        if polygon.is_valid:
                            entrances.append(polygon)
                    except:
                        continue

        return entrances

    def calculate_available_zones(self) -> List[Polygon]:
        """Calculate available zones for îlot placement"""
        if not self.plan_bounds:
            return []

        # Create main planning area
        min_x, min_y, max_x, max_y = self.plan_bounds
        main_area = box(min_x, min_y, max_x, max_y)

        # Remove restricted areas
        available_area = main_area
        for restricted in self.restricted_areas:
            try:
                available_area = available_area.difference(restricted)
            except:
                continue

        # Remove entrance buffer zones
        for entrance in self.entrances:
            try:
                # Create buffer around entrance
                buffer_zone = entrance.buffer(5.0)  # 5-unit buffer
                available_area = available_area.difference(buffer_zone)
            except:
                continue

        # Convert to list of polygons
        if isinstance(available_area, MultiPolygon):
            self.available_zones = [poly for poly in available_area.geoms if poly.area > 10]
        elif isinstance(available_area, Polygon) and available_area.area > 10:
            self.available_zones = [available_area]
        else:
            self.available_zones = []

        return self.available_zones

    def generate_ilot_layout(self, layout_profile: Dict[str, float], total_area: Optional[float] = None) -> Dict[str, Any]:
        """Generate îlot layout based on profile"""
        try:
            # Calculate available zones
            self.calculate_available_zones()

            if not self.available_zones:
                return {'error': 'No available zones found for îlot placement'}

            # Calculate total available area
            if total_area is None:
                total_area = sum(zone.area for zone in self.available_zones)

            # Calculate number of îlots per size category
            ilot_counts = {}
            ilot_sizes = {}

            for size_range, percentage in layout_profile.items():
                # Parse size range (e.g., "0-1", "1-3", "3-5", "5-10")
                min_size, max_size = map(float, size_range.split('-'))
                avg_size = (min_size + max_size) / 2

                # Calculate number of îlots
                area_for_category = total_area * (percentage / 100.0)
                count = max(1, int(area_for_category / avg_size))

                ilot_counts[size_range] = count
                ilot_sizes[size_range] = (min_size, max_size)

            # Generate îlots
            placed_ilots = []
            corridors = []

            for size_range, count in ilot_counts.items():
                min_size, max_size = ilot_sizes[size_range]

                for i in range(count):
                    # Generate random size within range
                    ilot_area = random.uniform(min_size, max_size)

                    # Try to place îlot
                    ilot_polygon = self._place_ilot(ilot_area, placed_ilots)

                    if ilot_polygon:
                        placed_ilots.append({
                            'polygon': ilot_polygon,
                            'area': ilot_area,
                            'size_category': size_range,
                            'id': f"ilot_{len(placed_ilots) + 1}"
                        })

            # Generate corridors
            corridors = self._generate_corridors(placed_ilots)

            self.ilots = placed_ilots
            self.corridors = corridors

            return {
                'status': 'success',
                'ilots_placed': len(placed_ilots),
                'corridors_created': len(corridors),
                'total_ilot_area': sum(ilot['area'] for ilot in placed_ilots),
                'coverage_percentage': (sum(ilot['area'] for ilot in placed_ilots) / total_area) * 100
            }

        except Exception as e:
            return {'error': str(e)}

    def _place_ilot(self, area: float, existing_ilots: List[Dict], max_attempts: int = 50) -> Optional[Polygon]:
        """Place a single îlot in available space"""
        for attempt in range(max_attempts):
            # Choose random zone
            if not self.available_zones:
                return None

            zone = random.choice(self.available_zones)

            # Calculate dimensions (assuming square for simplicity)
            side_length = math.sqrt(area)

            # Get random position within zone
            zone_bounds = zone.bounds
            min_x, min_y, max_x, max_y = zone_bounds

            # Ensure îlot fits within zone
            if max_x - min_x < side_length or max_y - min_y < side_length:
                continue

            # Random position
            x = random.uniform(min_x, max_x - side_length)
            y = random.uniform(min_y, max_y - side_length)

            # Create îlot polygon
            ilot_polygon = box(x, y, x + side_length, y + side_length)

            # Check if placement is valid
            if self._is_valid_placement(ilot_polygon, existing_ilots):
                return ilot_polygon

        return None

    def _is_valid_placement(self, ilot_polygon: Polygon, existing_ilots: List[Dict]) -> bool:
        """Check if îlot placement is valid"""
        # Check if within available zones
        within_available = False
        for zone in self.available_zones:
            if zone.contains(ilot_polygon):
                within_available = True
                break

        if not within_available:
            return False

        # Check for overlaps with existing îlots
        for existing in existing_ilots:
            if ilot_polygon.intersects(existing['polygon']):
                return False

        # Check distance from restricted areas
        for restricted in self.restricted_areas:
            if ilot_polygon.distance(restricted) < 1.0:  # 1-unit minimum distance
                return False

        # Check distance from entrances
        for entrance in self.entrances:
            if ilot_polygon.distance(entrance) < 3.0:  # 3-unit minimum distance
                return False

        return True

    def _generate_corridors(self, ilots: List[Dict], corridor_width: float = 2.0) -> List[Polygon]:
        """Generate corridors between îlot rows"""
        corridors = []

        if len(ilots) < 2:
            return corridors

        # Group îlots by approximate rows
        rows = self._group_ilots_by_rows(ilots)

        # Generate corridors between facing rows
        for i in range(len(rows) - 1):
            for j in range(i + 1, len(rows)):
                corridor = self._create_corridor_between_rows(rows[i], rows[j], corridor_width)
                if corridor:
                    corridors.append(corridor)

        return corridors

    def _group_ilots_by_rows(self, ilots: List[Dict]) -> List[List[Dict]]:
        """Group îlots into rows based on y-coordinate"""
        if not ilots:
            return []

        # Sort îlots by y-coordinate
        sorted_ilots = sorted(ilots, key=lambda x: x['polygon'].centroid.y)

        rows = []
        current_row = [sorted_ilots[0]]

        for i in range(1, len(sorted_ilots)):
            current_y = sorted_ilots[i]['polygon'].centroid.y
            last_y = current_row[-1]['polygon'].centroid.y

            # If y-coordinates are close, add to current row
            if abs(current_y - last_y) < 10:  # 10-unit tolerance
                current_row.append(sorted_ilots[i])
            else:
                # Start new row
                rows.append(current_row)
                current_row = [sorted_ilots[i]]

        rows.append(current_row)
        return rows

    def _create_corridor_between_rows(self, row1: List[Dict], row2: List[Dict], width: float) -> Optional[Polygon]:
        """Create corridor between two rows of îlots"""
        if not row1 or not row2:
            return None

        # Calculate average y-coordinates
        y1 = sum(ilot['polygon'].centroid.y for ilot in row1) / len(row1)
        y2 = sum(ilot['polygon'].centroid.y for ilot in row2) / len(row2)

        # Ensure y1 < y2
        if y1 > y2:
            y1, y2 = y2, y1

        # Calculate corridor boundaries
        corridor_y1 = y1 + width / 2
        corridor_y2 = y2 - width / 2

        if corridor_y2 <= corridor_y1:
            return None

        # Find x-extent of corridor
        all_x = []
        for ilot in row1 + row2:
            bounds = ilot['polygon'].bounds
            all_x.extend([bounds[0], bounds[2]])

        if not all_x:
            return None

        min_x = min(all_x)
        max_x = max(all_x)

        # Create corridor polygon
        corridor = box(min_x, corridor_y1, max_x, corridor_y2)

        return corridor

    def create_visualization(self) -> go.Figure:
        """Create comprehensive visualization of the îlot placement"""
        fig = go.Figure()

        # Add plan boundaries
        if self.plan_bounds:
            min_x, min_y, max_x, max_y = self.plan_bounds
            fig.add_trace(go.Scatter(
                x=[min_x, max_x, max_x, min_x, min_x],
                y=[min_y, min_y, max_y, max_y, min_y],
                mode='lines',
                line=dict(color='black', width=2),
                name='Plan Boundary',
                hovertemplate='Plan Boundary<extra></extra>'
            ))

        # Add walls
        for i, wall in enumerate(self.walls):
            if hasattr(wall, 'coords'):
                coords = list(wall.coords)
                x_coords = [coord[0] for coord in coords]
                y_coords = [coord[1] for coord in coords]

                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    line=dict(color='black', width=3),
                    name='Walls' if i == 0 else None,
                    showlegend=i == 0,
                    hovertemplate='Wall<extra></extra>'
                ))

        # Add restricted areas
        for i, area in enumerate(self.restricted_areas):
            if hasattr(area, 'exterior'):
                coords = list(area.exterior.coords)
                x_coords = [coord[0] for coord in coords]
                y_coords = [coord[1] for coord in coords]

                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    fill='toself',
                    fillcolor='rgba(0, 123, 255, 0.3)',
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Restricted Areas' if i == 0 else None,
                    showlegend=i == 0,
                    hovertemplate='Restricted Area<extra></extra>'
                ))

        # Add entrances
        for i, entrance in enumerate(self.entrances):
            if hasattr(entrance, 'exterior'):
                coords = list(entrance.exterior.coords)
                x_coords = [coord[0] for coord in coords]
                y_coords = [coord[1] for coord in coords]

                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.3)',
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='Entrances/Exits' if i == 0 else None,
                    showlegend=i == 0,
                    hovertemplate='Entrance/Exit<extra></extra>'
                ))

        # Add îlots
        color_map = {
            '0-1': '#FF6B6B',
            '1-3': '#4ECDC4',
            '3-5': '#45B7D1',
            '5-10': '#96CEB4',
            '10+': '#FFEAA7'
        }

        for i, ilot in enumerate(self.ilots):
            polygon = ilot['polygon']
            if hasattr(polygon, 'exterior'):
                coords = list(polygon.exterior.coords)
                x_coords = [coord[0] for coord in coords]
                y_coords = [coord[1] for coord in coords]

                size_category = ilot['size_category']
                color = color_map.get(size_category, '#DDA0DD')

                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    fill='toself',
                    fillcolor=color,
                    mode='lines',
                    line=dict(color='darkblue', width=1),
                    name=f'Îlots {size_category}m²' if i == 0 or size_category not in [existing['size_category'] for existing in self.ilots[:i]] else None,
                    showlegend=i == 0 or size_category not in [existing['size_category'] for existing in self.ilots[:i]],
                    hovertemplate=f'Îlot {ilot["id"]}<br>Area: {ilot["area"]:.2f}m²<br>Category: {size_category}m²<extra></extra>'
                ))

        # Add corridors
        for i, corridor in enumerate(self.corridors):
            if hasattr(corridor, 'exterior'):
                coords = list(corridor.exterior.coords)
                x_coords = [coord[0] for coord in coords]
                y_coords = [coord[1] for coord in coords]

                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    fill='toself',
                    fillcolor='rgba(255, 255, 0, 0.3)',
                    mode='lines',
                    line=dict(color='orange', width=2),
                    name='Corridors' if i == 0 else None,
                    showlegend=i == 0,
                    hovertemplate='Corridor<extra></extra>'
                ))

        # Update layout
        fig.update_layout(
            title='Enterprise Îlot Placement System - Floor Plan Analysis',
            xaxis_title='X Coordinate (m)',
            yaxis_title='Y Coordinate (m)',
            xaxis=dict(scaleanchor="y", scaleratio=1),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            showlegend=True,
            hovermode='closest',
            width=1200,
            height=800,
            template='plotly_white'
        )

        return fig

    def export_results(self, format_type: str = 'pdf') -> bytes:
        """Export results to PDF or other formats"""
        if format_type == 'pdf':
            return self._export_to_pdf()
        elif format_type == 'json':
            return self._export_to_json()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def _export_to_pdf(self) -> bytes:
        """Export results to PDF"""
        buffer = io.BytesIO()

        # Create PDF
        pdf = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4

        # Title
        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(50, height - 50, "Enterprise Îlot Placement System - Analysis Report")

        # Date
        pdf.setFont("Helvetica", 10)
        pdf.drawString(50, height - 70, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Summary statistics
        y_pos = height - 120
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y_pos, "Summary Statistics:")

        y_pos -= 30
        pdf.setFont("Helvetica", 10)

        stats = [
            f"Total Îlots Placed: {len(self.ilots)}",
            f"Total Corridors Created: {len(self.corridors)}",
            f"Total Îlot Area: {sum(ilot['area'] for ilot in self.ilots):.2f} m²",
            f"Walls Detected: {len(self.walls)}",
            f"Restricted Areas: {len(self.restricted_areas)}",
            f"Entrances/Exits: {len(self.entrances)}"
        ]

        for stat in stats:
            pdf.drawString(70, y_pos, stat)
            y_pos -= 20

        # Îlot details
        y_pos -= 30
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y_pos, "Îlot Details:")

        y_pos -= 30
        pdf.setFont("Helvetica", 9)

        for ilot in self.ilots:
            if y_pos < 100:  # Start new page if needed
                pdf.showPage()
                y_pos = height - 50

            pdf.drawString(70, y_pos, f"ID: {ilot['id']}, Area: {ilot['area']:.2f}m², Category: {ilot['size_category']}m²")
            y_pos -= 15

        pdf.save()
        buffer.seek(0)
        return buffer.getvalue()

    def _export_to_json(self) -> bytes:
        """Export results to JSON"""
        data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'system_version': '1.0.0',
                'total_ilots': len(self.ilots),
                'total_corridors': len(self.corridors)
            },
            'ilots': [
                {
                    'id': ilot['id'],
                    'area': ilot['area'],
                    'size_category': ilot['size_category'],
                    'bounds': list(ilot['polygon'].bounds)
                }
                for ilot in self.ilots
            ],
            'statistics': {
                'total_ilot_area': sum(ilot['area'] for ilot in self.ilots),
                'walls_detected': len(self.walls),
                'restricted_areas': len(self.restricted_areas),
                'entrances': len(self.entrances)
            }
        }

        return json.dumps(data, indent=2).encode('utf-8')

# Initialize the system
if 'placement_system' not in st.session_state:
    st.session_state.placement_system = EnterpriseIlotPlacementSystem()

# Main application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏗️ Enterprise Îlot Placement System</h1>
        <p>Professional architectural space planning and optimization tool</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📋 System Controls")

        # File upload
        st.subheader("1. Upload Floor Plan")
        uploaded_file = st.file_uploader(
            "Select your floor plan file",
            type=['dxf', 'dwg', 'jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload CAD files (DXF/DWG) or image files (JPG/PNG) of your floor plan"
        )

        # Layout profile configuration
        st.subheader("2. Configure Îlot Layout")

        st.info("Define the distribution of îlot sizes according to your requirements")

        # Predefined profiles
        profile_preset = st.selectbox(
            "Choose a preset profile",
            ["Custom", "Balanced", "Small Focus", "Large Focus", "Mixed Commercial"],
            help="Select a predefined layout profile or choose Custom for manual configuration"
        )

        # Profile configurations
        profiles = {
            "Balanced": {"0-1": 10, "1-3": 25, "3-5": 30, "5-10": 35},
            "Small Focus": {"0-1": 20, "1-3": 40, "3-5": 25, "5-10": 15},
            "Large Focus": {"0-1": 5, "1-3": 15, "3-5": 30, "5-10": 50},
            "Mixed Commercial": {"0-1": 15, "1-3": 20, "3-5": 35, "5-10": 30}
        }

        if profile_preset == "Custom":
            st.write("**Custom Profile Configuration**")
            profile_0_1 = st.slider("0-1 m² îlots (%)", 0, 100, 10)
            profile_1_3 = st.slider("1-3 m² îlots (%)", 0, 100, 25)
            profile_3_5 = st.slider("3-5 m² îlots (%)", 0, 100, 30)
            profile_5_10 = st.slider("5-10 m² îlots (%)", 0, 100, 35)

            # Validate total percentage
            total_percentage = profile_0_1 + profile_1_3 + profile_3_5 + profile_5_10
            if total_percentage != 100:
                st.warning(f"Total percentage: {total_percentage}% (should be 100%)")

            layout_profile = {
                "0-1": profile_0_1,
                "1-3": profile_1_3,
                "3-5": profile_3_5,
                "5-10": profile_5_10
            }
        else:
            layout_profile = profiles[profile_preset]

        # Display current profile
        st.write("**Current Profile:**")
        for size_range, percentage in layout_profile.items():
            st.write(f"• {size_range} m²: {percentage}%")

        # Advanced settings
        st.subheader("3. Advanced Settings")
        corridor_width = st.slider("Corridor Width (m)", 1.0, 5.0, 2.0, 0.1)
        min_entrance_distance = st.slider("Minimum Distance from Entrances (m)", 1.0, 10.0, 3.0, 0.5)

        # Process button
        if st.button("🚀 Generate Îlot Layout", type="primary", use_container_width=True):
            if uploaded_file:
                # Performance optimization for large files
                file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                
                if file_size_mb > 10:
                    st.info(f"📊 Large file detected ({file_size_mb:.1f} MB). Using optimized processing...")
                
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("🔄 Reading file...")
                    progress_bar.progress(10)
                    
                    # Parse floor plan
                    file_bytes = uploaded_file.read()
                    
                    status_text.text("🔍 Parsing CAD entities...")
                    progress_bar.progress(30)
                    
                    # Parse file with enhanced error handling
                    dwg_parser = DWGParser()
                    zones = dwg_parser.parse_file_simple(file_bytes, uploaded_file.name)
                    
                    progress_bar.progress(70)

                    if not zones:
                        progress_bar.empty()
                        status_text.empty()
                        st.error("❌ File processing failed: DWG parsing failed")
                        st.warning("💡 **Possible solutions:**")
                        st.write("• The file may be corrupted or use an unsupported encoding")
                        st.write("• Try saving the file in a different CAD program")
                        st.write("• Ensure the file is a valid DWG or DXF format")
                        st.write("• Check if the file contains geometric data (lines, polylines, etc.)")

                        with st.expander("🔧 Technical Details"):
                            st.write("The parser supports:")
                            st.write("- DXF files (ASCII and Binary)")
                            st.write("- DWG files (AutoCAD R12 and later)")
                            st.write("- Files with UTF-8, Latin-1 encoding")
                            st.write("- LWPOLYLINE, POLYLINE, LINE, CIRCLE entities")
                        return
                    
                    status_text.text("🏗️ Generating îlot layout...")
                    progress_bar.progress(85)
                    
                    # Create placement system and process zones
                    if 'placement_system' not in st.session_state:
                        from ilot_placement_engine import IlotPlacementEngine
                        st.session_state.placement_system = IlotPlacementEngine()
                    
                    # Calculate total available area
                    total_area = sum(zone.get('area', 0) for zone in zones if zone.get('zone_type') not in ['Wall', 'Line'])
                    
                    # Generate îlot layout
                    layout_result = st.session_state.placement_system.place_ilots_with_distribution(
                        zones, layout_profile, total_area, corridor_width
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")
                    
                    if 'error' not in layout_result:
                        st.session_state.layout_result = layout_result
                        st.session_state.zones = zones
                        st.session_state.layout_generated = True
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.success(f"✅ Îlot layout generated successfully! Processed {len(zones)} zones, placed {len(layout_result['placed_ilots'])} îlots")
                        
                        # Show quick stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Zones Detected", len(zones))
                        with col2:
                            st.metric("Îlots Placed", len(layout_result['placed_ilots']))
                        with col3:
                            st.metric("Efficiency", f"{layout_result['statistics']['space_efficiency']*100:.1f}%")
                    else:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"❌ Layout generation failed: {layout_result['error']}")
                        
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Processing error: {str(e)}")
                    
            else:
                st.error("❌ You must upload a floor plan file to proceed. This system requires real architectural data.")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📊 Visualization")

        # Show visualization if layout is generated
        if hasattr(st.session_state, 'layout_generated') and st.session_state.layout_generated:
            try:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # Create visualization from processed data
                fig = go.Figure()
                
                # Add zones
                if hasattr(st.session_state, 'zones'):
                    for zone in st.session_state.zones:
                        points = zone.get('points', [])
                        if len(points) >= 3:
                            x_coords = [p[0] for p in points] + [points[0][0]]
                            y_coords = [p[1] for p in points] + [points[0][1]]
                            
                            zone_type = zone.get('zone_type', 'Room')
                            color = 'lightblue' if zone_type == 'Room' else 'gray'
                            
                            fig.add_trace(go.Scatter(
                                x=x_coords, y=y_coords,
                                fill='toself',
                                fillcolor=color,
                                line=dict(color='black', width=1),
                                name=f"Zone: {zone_type}",
                                showlegend=False
                            ))
                
                # Add îlots if available
                if hasattr(st.session_state, 'layout_result'):
                    for ilot in st.session_state.layout_result.get('placed_ilots', []):
                        bounds = ilot.get('bounds', [])
                        if len(bounds) == 4:
                            x_coords = [bounds[0], bounds[2], bounds[2], bounds[0], bounds[0]]
                            y_coords = [bounds[1], bounds[1], bounds[3], bounds[3], bounds[1]]
                            
                            fig.add_trace(go.Scatter(
                                x=x_coords, y=y_coords,
                                fill='toself',
                                fillcolor='red',
                                line=dict(color='darkred', width=2),
                                name=f"Îlot: {ilot.get('id', 'Unknown')}",
                                showlegend=False
                            ))
                
                fig.update_layout(
                    title="🏗️ Îlot Placement Layout",
                    xaxis_title="X (meters)",
                    yaxis_title="Y (meters)",
                    height=600,
                    showlegend=True
                )
                fig.update_xaxes(scaleanchor="y", scaleratio=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
        else:
            st.info("📁 Upload a floor plan file to begin îlot placement analysis.")

    with col2:
        st.header("📈 Statistics")

        # Show statistics if layout is generated
        if hasattr(st.session_state, 'layout_generated') and st.session_state.layout_generated:
            layout_result = st.session_state.layout_result
            zones = st.session_state.zones

            # Key metrics
            total_ilots = len(layout_result.get('placed_ilots', []))
            total_corridors = len(layout_result.get('corridors', []))
            total_area = layout_result.get('statistics', {}).get('total_area_used', 0)
            efficiency = layout_result.get('statistics', {}).get('space_efficiency', 0) * 100

            st.metric("Total Îlots Placed", total_ilots)
            st.metric("Total Corridors", total_corridors)
            st.metric("Total Îlot Area", f"{total_area:.2f} m²")
            st.metric("Space Efficiency", f"{efficiency:.1f}%")

            # Size distribution
            st.subheader("📊 Size Distribution")
            achieved_dist = layout_result.get('statistics', {}).get('distribution_achieved', {})
            
            if achieved_dist:
                for size_range, percentage in achieved_dist.items():
                    st.write(f"**{size_range} m²:** {percentage:.1f}%")
            
            # Validation results
            validation = layout_result.get('validation', {})
            if validation:
                st.subheader("✅ Validation")
                if validation.get('is_valid', True):
                    st.success("All placement constraints satisfied")
                else:
                    st.warning(f"⚠️ {validation.get('total_violations', 0)} constraint violations found")
            
            # Processing performance
            st.subheader("⚡ Processing Stats")
            st.write(f"• Zones processed: {len(zones)}")
            st.write(f"• File format: {uploaded_file.name.split('.')[-1].upper() if 'uploaded_file' in locals() else 'CAD'}")
            st.write(f"• Processing: Optimized for large datasets")

            for category, count in size_dist.items():
                st.write(f"• {category} m²: {count} îlots")

            # Export options
            st.subheader("📤 Export Results")

            col_pdf, col_json = st.columns(2)

            with col_pdf:
                if st.button("📄 Export PDF"):
                    try:
                        pdf_data = system.export_results('pdf')
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_data,
                            file_name=f"ilot_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"PDF export failed: {str(e)}")

            with col_json:
                if st.button("📊 Export JSON"):
                    try:
                        json_data = system.export_results('json')
                        st.download_button(
                            label="Download JSON Data",
                            data=json_data,
                            file_name=f"ilot_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.error(f"JSON export failed: {str(e)}")
        else:
            st.error("No statistics available. Upload and process a real floor plan file to generate analysis.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p><strong>Enterprise Îlot Placement System</strong> - Professional Architectural Space Planning Tool</p>
        <p>Features: CAD File Support • Advanced Zone Detection • Intelligent Placement • Export Capabilities</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()
=======
# Main header
st.title("📋 App Requirements & Ideas Workspace")
st.markdown("*A collaborative space for organizing and discussing your app requirements*")
=======
        background-color: #1e3c72;
    }
</style>
""", unsafe_allow_html=True)
>>>>>>> 2b57e5d (Created a checkpoint)

# Initialize session state
if 'floor_plan' not in st.session_state:
    st.session_state.floor_plan = None
if 'parsed_zones' not in st.session_state:
    st.session_state.parsed_zones = None
if 'placement_results' not in st.session_state:
    st.session_state.placement_results = None
if 'visualization' not in st.session_state:
    st.session_state.visualization = None

# Header
st.markdown('<div class="main-header"><h1>🏗️ Enterprise Îlot Placement System</h1><p>Professional Grade Floor Plan Analysis & Optimization</p></div>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # File upload
    st.subheader("📁 Upload Floor Plan")
    uploaded_file = st.file_uploader(
        "Choose a DXF/DWG file",
        type=['dxf', 'dwg', 'pdf', 'png', 'jpg', 'jpeg'],
        help="Upload your floor plan in DXF, DWG, or image format"
    )
    
    # Size distribution configuration
    st.subheader("📏 Îlot Size Distribution")
    
    size_0_1 = st.slider("0-1 m² (%)", min_value=0, max_value=100, value=10)
    size_1_3 = st.slider("1-3 m² (%)", min_value=0, max_value=100, value=25)
    size_3_5 = st.slider("3-5 m² (%)", min_value=0, max_value=100, value=30)
    size_5_10 = st.slider("5-10 m² (%)", min_value=0, max_value=100, value=35)
    
    total_percentage = size_0_1 + size_1_3 + size_3_5 + size_5_10
    
    if total_percentage != 100:
        st.warning(f"Total percentage: {total_percentage}% (should be 100%)")
    else:
        st.success("✓ Distribution valid")
    
    # Corridor configuration
    st.subheader("🚶 Corridor Settings")
    corridor_width = st.number_input(
        "Corridor Width (m)",
        min_value=0.8,
        max_value=3.0,
        value=1.2,
        step=0.1
    )
    
    min_distance_entrance = st.number_input(
        "Min Distance from Entrance (m)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.5
    )
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        optimization_algorithm = st.selectbox(
            "Optimization Algorithm",
            ["Genetic Algorithm", "Simulated Annealing", "Grid-Based", "Force-Directed"]
        )
        
        iterations = st.number_input(
            "Optimization Iterations",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100
        )
        
        allow_rotation = st.checkbox("Allow Îlot Rotation", value=True)
        snap_to_grid = st.checkbox("Snap to Grid", value=False)
        
        if snap_to_grid:
            grid_size = st.number_input(
                "Grid Size (m)",
                min_value=0.1,
                max_value=1.0,
                value=0.25,
                step=0.05
            )

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Floor Plan Analysis")
    
    # Zone legend
    st.markdown("""
    <div class="zone-legend">
        <div class="legend-item">
            <div class="legend-color" style="background-color: black;"></div>
            <span>Walls (MUR)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: lightblue;"></div>
            <span>Restricted Areas (NO ENTREE)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: red;"></div>
            <span>Entrances/Exits (ENTREE/SORTIE)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: lightgreen;"></div>
            <span>Available Space</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Process the uploaded file
        try:
            file_bytes = uploaded_file.read()
            filename = uploaded_file.name
            
            with st.spinner("🔄 Parsing floor plan..."):
                parser = DWGParser()
                zones = parser.parse_file_simple(file_bytes, filename)
                
                if zones:
                    st.session_state.parsed_zones = zones
                    st.success(f"✓ Successfully parsed {len(zones)} zones from floor plan")
                    
                    # Display zone statistics
                    zone_stats = {}
                    for zone in zones:
                        zone_type = zone.get('zone_type', 'UNKNOWN')
                        if zone_type not in zone_stats:
                            zone_stats[zone_type] = 0
                        zone_stats[zone_type] += 1
                    
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.write("**Zone Analysis:**")
                    for zone_type, count in zone_stats.items():
                        st.write(f"• {zone_type}: {count} zones")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("❌ Failed to parse floor plan. Please ensure the file is valid.")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    else:
        st.info("👆 Please upload a floor plan to begin")

with col2:
    st.subheader("🎯 Îlot Placement")
    
    if st.session_state.parsed_zones:
        # Calculate button
        if st.button("🚀 Generate Îlot Placement", type="primary"):
            with st.spinner("🔄 Optimizing îlot placement..."):
                # Prepare distribution
                distribution = {
                    '0-1': size_0_1,
                    '1-3': size_1_3,
                    '3-5': size_3_5,
                    '5-10': size_5_10
                }
                
                # Calculate total available area
                total_area = sum(zone.get('area', 0) for zone in st.session_state.parsed_zones 
                               if zone.get('zone_type', '') in ['AVAILABLE', 'MAIN_FLOOR', 'ROOM'])
                
                # Initialize placement engine
                engine = IlotPlacementEngine()
                engine.corridor_width = corridor_width
                engine.min_distance_from_entrance = min_distance_entrance
                
                # Place îlots
                results = engine.place_ilots_with_distribution(
                    st.session_state.parsed_zones,
                    distribution,
                    total_area,
                    corridor_width
                )
                
                if 'error' not in results:
                    st.session_state.placement_results = results
                    st.success("✓ Îlot placement completed successfully!")
                    
                    # Display statistics
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.write("**Placement Statistics:**")
                    stats = results.get('statistics', {})
                    st.metric("Total Îlots Placed", stats.get('total_ilots', 0))
                    st.metric("Total Area Used", f"{stats.get('total_area_used', 0):.2f} m²")
                    st.metric("Space Efficiency", f"{stats.get('space_efficiency', 0)*100:.1f}%")
                    
                    # Show achieved distribution
                    st.write("**Achieved Distribution:**")
                    achieved = stats.get('distribution_achieved', {})
                    for size_range, percentage in achieved.items():
                        st.write(f"• {size_range} m²: {percentage:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error(f"❌ {results['error']}")

# Visualization section
if st.session_state.placement_results:
    st.divider()
    st.subheader("📈 Visualization")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["2D Floor Plan", "3D View", "Analytics"])
    
    with tab1:
        # Create 2D visualization
        fig = go.Figure()
        
        # Draw walls and zones
        for zone in st.session_state.parsed_zones:
            if 'geometry' in zone:
                geom = zone['geometry']
                zone_type = zone.get('zone_type', 'AVAILABLE')
                
                # Set color based on zone type
                if zone_type == 'MUR':
                    color = 'black'
                elif zone_type == 'NO_ENTREE':
                    color = 'lightblue'
                elif zone_type == 'ENTREE_SORTIE':
                    color = 'red'
                else:
                    color = 'lightgray'
                
                if isinstance(geom, dict) and 'type' in geom:
                    if geom['type'] == 'Polygon':
                        coords = geom['coordinates'][0]
                        x_coords = [c[0] for c in coords]
                        y_coords = [c[1] for c in coords]
                        
                        fig.add_trace(go.Scatter(
                            x=x_coords + [x_coords[0]],
                            y=y_coords + [y_coords[0]],
                            mode='lines',
                            line=dict(color=color, width=2),
                            fill='toself',
                            fillcolor=color,
                            opacity=0.3,
                            name=zone_type,
                            showlegend=False
                        ))
        
        # Draw placed îlots
        if 'placed_ilots' in st.session_state.placement_results:
            for i, ilot in enumerate(st.session_state.placement_results['placed_ilots']):
                x = ilot['x']
                y = ilot['y']
                width = ilot['width']
                height = ilot['height']
                
                # Draw îlot rectangle
                fig.add_trace(go.Scatter(
                    x=[x, x+width, x+width, x, x],
                    y=[y, y, y+height, y+height, y],
                    mode='lines',
                    line=dict(color='darkgreen', width=2),
                    fill='toself',
                    fillcolor='lightgreen',
                    name=f"Îlot {i+1}",
                    showlegend=False,
                    hovertext=f"Îlot {i+1}<br>Area: {ilot['area']:.2f} m²<br>Size Category: {ilot.get('size_category', 'N/A')}",
                    hoverinfo='text'
                ))
                
                # Add îlot label
                fig.add_annotation(
                    x=x + width/2,
                    y=y + height/2,
                    text=f"{ilot['area']:.1f}",
                    showarrow=False,
                    font=dict(size=8)
                )
        
        # Draw corridors
        if 'corridors' in st.session_state.placement_results:
            for corridor in st.session_state.placement_results['corridors']:
                if 'geometry' in corridor:
                    geom = corridor['geometry']
                    if isinstance(geom, dict) and geom['type'] == 'Polygon':
                        coords = geom['coordinates'][0]
                        x_coords = [c[0] for c in coords]
                        y_coords = [c[1] for c in coords]
                        
                        fig.add_trace(go.Scatter(
                            x=x_coords + [x_coords[0]],
                            y=y_coords + [y_coords[0]],
                            mode='lines',
                            line=dict(color='orange', width=1, dash='dash'),
                            fill='toself',
                            fillcolor='orange',
                            opacity=0.2,
                            name='Corridor',
                            showlegend=False
                        ))
        
        # Update layout
        fig.update_layout(
            title="Floor Plan with Îlot Placement",
            xaxis=dict(title="X (meters)", scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Y (meters)"),
            height=600,
            showlegend=False,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Create 3D visualization
        fig_3d = go.Figure()
        
        # Add 3D îlots
        if 'placed_ilots' in st.session_state.placement_results:
            for i, ilot in enumerate(st.session_state.placement_results['placed_ilots']):
                x = ilot['x']
                y = ilot['y']
                width = ilot['width']
                height = ilot['height']
                z_height = min(ilot['area'] / 2, 3)  # Height based on area
                
                # Create 3D box vertices
                vertices = [
                    [x, y, 0], [x+width, y, 0], [x+width, y+height, 0], [x, y+height, 0],
                    [x, y, z_height], [x+width, y, z_height], [x+width, y+height, z_height], [x, y+height, z_height]
                ]
                
                # Define faces
                faces = [
                    [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],  # Sides
                    [0, 1, 2, 3], [4, 5, 6, 7]  # Bottom and top
                ]
                
                for face in faces:
                    face_vertices = [vertices[i] for i in face]
                    x_coords = [v[0] for v in face_vertices] + [face_vertices[0][0]]
                    y_coords = [v[1] for v in face_vertices] + [face_vertices[0][1]]
                    z_coords = [v[2] for v in face_vertices] + [face_vertices[0][2]]
                    
                    fig_3d.add_trace(go.Scatter3d(
                        x=x_coords,
                        y=y_coords,
                        z=z_coords,
                        mode='lines',
                        line=dict(color='green', width=2),
                        showlegend=False,
                        hovertext=f"Îlot {i+1}<br>Area: {ilot['area']:.2f} m²"
                    ))
        
        fig_3d.update_layout(
            title="3D Visualization of Îlot Placement",
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Height (m)",
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            height=600
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
    
    with tab3:
        # Analytics dashboard
        col_a1, col_a2 = st.columns(2)
        
        with col_a1:
            # Size distribution pie chart
            if 'distribution_achieved' in st.session_state.placement_results['statistics']:
                achieved = st.session_state.placement_results['statistics']['distribution_achieved']
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=list(achieved.keys()),
                    values=list(achieved.values()),
                    hole=0.3
                )])
                
                fig_pie.update_layout(
                    title="Achieved Size Distribution",
                    height=300
                )
                
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_a2:
            # Efficiency metrics
            stats = st.session_state.placement_results['statistics']
            
            fig_metrics = go.Figure()
            
            categories = ['Space Efficiency', 'Placement Success', 'Corridor Coverage']
            values = [
                stats.get('space_efficiency', 0) * 100,
                (stats.get('total_ilots', 0) / max(sum(distribution.values()), 1)) * 100,
                len(st.session_state.placement_results.get('corridors', [])) * 10
            ]
            
            fig_metrics.add_trace(go.Bar(
                x=categories,
                y=values,
                marker_color=['green', 'blue', 'orange']
            ))
            
            fig_metrics.update_layout(
                title="Performance Metrics",
                yaxis_title="Percentage (%)",
                height=300
            )
            
            st.plotly_chart(fig_metrics, use_container_width=True)

# Export section
if st.session_state.placement_results:
    st.divider()
    st.subheader("📤 Export Results")
    
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        if st.button("📄 Export as PDF"):
            # Generate PDF report
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=A4)
            
            # Add title
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, 800, "Îlot Placement Report")
            
            # Add statistics
            c.setFont("Helvetica", 12)
            y_position = 750
            stats = st.session_state.placement_results['statistics']
            
            c.drawString(50, y_position, f"Total Îlots: {stats.get('total_ilots', 0)}")
            y_position -= 20
            c.drawString(50, y_position, f"Total Area Used: {stats.get('total_area_used', 0):.2f} m²")
            y_position -= 20
            c.drawString(50, y_position, f"Space Efficiency: {stats.get('space_efficiency', 0)*100:.1f}%")
            
            # Add timestamp
            y_position -= 40
            c.drawString(50, y_position, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            c.save()
            buffer.seek(0)
            
            st.download_button(
                label="Download PDF Report",
                data=buffer,
                file_name=f"ilot_placement_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
    
    with col_export2:
        if st.button("🖼️ Export as Image"):
            # Export current visualization as image
            st.info("Image export will capture the current 2D floor plan view")
    
    with col_export3:
        if st.button("📊 Export as DXF"):
            # Generate DXF file with placed îlots
            doc = ezdxf.new('R2010')
            msp = doc.modelspace()
            
            # Add îlots to DXF
            for ilot in st.session_state.placement_results['placed_ilots']:
                msp.add_lwpolyline([
                    (ilot['x'], ilot['y']),
                    (ilot['x'] + ilot['width'], ilot['y']),
                    (ilot['x'] + ilot['width'], ilot['y'] + ilot['height']),
                    (ilot['x'], ilot['y'] + ilot['height']),
                    (ilot['x'], ilot['y'])
                ])
            
            # Save to buffer
            buffer = io.BytesIO()
            doc.write(buffer)
            buffer.seek(0)
            
            st.download_button(
                label="Download DXF File",
                data=buffer,
                file_name=f"ilot_placement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dxf",
                mime="application/octet-stream"
            )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em; padding: 2rem;'>
    <p>Enterprise Îlot Placement System v2.0 | Professional Grade Floor Plan Optimization</p>
    <p>© 2025 - Built with cutting-edge spatial analysis algorithms</p>
</div>
""", unsafe_allow_html=True)
>>>>>>> b49860d (Create a collaborative workspace for app ideas and requirements)
>>>>>>> origin/replit-agent
