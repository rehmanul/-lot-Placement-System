"""
Real DWG/DXF Parser - No Fallback Version
Only processes actual DWG/DXF files with real geometric data
"""

import tempfile
import os
from typing import List, Dict, Any, Optional
import traceback

try:
    import ezdxf
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False

class DWGParser:
    def __init__(self):
        self.supported_formats = ['.dwg', '.dxf']

    def parse_file_simple(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Parse DWG/DXF file - REAL VERSION ONLY
        Returns empty list if file cannot be parsed as real DWG/DXF
        """
        if not EZDXF_AVAILABLE:
            print("ERROR: ezdxf not available - cannot parse real DWG/DXF files")
            return []

        # Only process actual DWG/DXF files
        file_ext = os.path.splitext(filename.lower())[1]
        if file_ext not in self.supported_formats:
            print(f"ERROR: File '{filename}' is not a supported DWG/DXF format")
            return []

        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
                temp_file.write(file_bytes)
                temp_file_path = temp_file.name

            try:
                # Parse real DWG/DXF file
                zones = self._parse_real_dwg_dxf(temp_file_path, filename)

                if not zones:
                    print(f"ERROR: No valid geometric data found in '{filename}'")
                    return []

                print(f"SUCCESS: Parsed {len(zones)} real zones from '{filename}'")
                return zones

            finally:
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except Exception as e:
            print(f"ERROR: Failed to parse '{filename}': {str(e)}")
            traceback.print_exc()
            return []

    def _parse_real_dwg_dxf(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Parse actual DWG/DXF file using ezdxf"""
        try:
            # Try to read as DXF first
            try:
                doc = ezdxf.readfile(file_path)
                print(f"Successfully opened as DXF: {filename}")
            except ezdxf.DXFStructureError:
                # If DXF fails, try DWG
                try:
                    doc = ezdxf.readfile(file_path)
                    print(f"Successfully opened as DWG: {filename}")
                except Exception as e:
                    print(f"ERROR: Cannot open '{filename}' as DWG/DXF: {str(e)}")
                    return []

            # Extract real geometric entities
            zones = []
            msp = doc.modelspace()

            # Process lines, polylines, lwpolylines, circles, arcs
            entities_processed = 0

            # Process LWPOLYLINE entities (most common for room boundaries)
            for entity in msp.query('LWPOLYLINE'):
                try:
                    points = list(entity.vertices_in_wcs())
                    if len(points) >= 3:
                        zone = {
                            'zone_id': f"lwpoly_{entities_processed}",
                            'zone_type': self._classify_entity_type(entity),
                            'points': [(float(p.x), float(p.y)) for p in points],
                            'area': self._calculate_polygon_area(points),
                            'layer': entity.dxf.layer if hasattr(entity.dxf, 'layer') else 'Unknown',
                            'entity_type': 'LWPOLYLINE'
                        }
                        zones.append(zone)
                        entities_processed += 1
                except Exception as e:
                    print(f"Warning: Error processing LWPOLYLINE: {str(e)}")
                    continue

            # Process POLYLINE entities
            for entity in msp.query('POLYLINE'):
                try:
                    points = list(entity.vertices_in_wcs())
                    if len(points) >= 3:
                        zone = {
                            'zone_id': f"poly_{entities_processed}",
                            'zone_type': self._classify_entity_type(entity),
                            'points': [(float(p.x), float(p.y)) for p in points],
                            'area': self._calculate_polygon_area(points),
                            'layer': entity.dxf.layer if hasattr(entity.dxf, 'layer') else 'Unknown',
                            'entity_type': 'POLYLINE'
                        }
                        zones.append(zone)
                        entities_processed += 1
                except Exception as e:
                    print(f"Warning: Error processing POLYLINE: {str(e)}")
                    continue

            # Process CIRCLE entities
            for entity in msp.query('CIRCLE'):
                try:
                    center = entity.dxf.center
                    radius = entity.dxf.radius
                    # Create circular polygon approximation
                    points = self._create_circle_points(center, radius)
                    zone = {
                        'zone_id': f"circle_{entities_processed}",
                        'zone_type': self._classify_entity_type(entity),
                        'points': points,
                        'area': 3.14159 * radius * radius,
                        'layer': entity.dxf.layer if hasattr(entity.dxf, 'layer') else 'Unknown',
                        'entity_type': 'CIRCLE'
                    }
                    zones.append(zone)
                    entities_processed += 1
                except Exception as e:
                    print(f"Warning: Error processing CIRCLE: {str(e)}")
                    continue

            # Process LINE entities and try to form closed polygons
            lines = list(msp.query('LINE'))
            if lines:
                connected_polygons = self._connect_lines_to_polygons(lines)
                for i, polygon in enumerate(connected_polygons):
                    zone = {
                        'zone_id': f"line_polygon_{entities_processed}",
                        'zone_type': 'Room',
                        'points': polygon,
                        'area': self._calculate_polygon_area_coords(polygon),
                        'layer': 'LINE_ASSEMBLY',
                        'entity_type': 'LINE_POLYGON'
                    }
                    zones.append(zone)
                    entities_processed += 1

            print(f"Real parsing result: {entities_processed} entities processed, {len(zones)} zones created")
            return zones

        except Exception as e:
            print(f"ERROR: Real DWG/DXF parsing failed: {str(e)}")
            traceback.print_exc()
            return []

    def _classify_entity_type(self, entity) -> str:
        """Classify entity type based on layer name and properties"""
        layer_name = entity.dxf.layer.lower() if hasattr(entity.dxf, 'layer') else ''

        if 'wall' in layer_name or 'mur' in layer_name:
            return 'Wall'
        elif 'door' in layer_name or 'porte' in layer_name:
            return 'Door'
        elif 'window' in layer_name or 'fenetre' in layer_name:
            return 'Window'
        elif 'kitchen' in layer_name or 'cuisine' in layer_name:
            return 'Kitchen'
        elif 'bath' in layer_name or 'salle' in layer_name:
            return 'Bathroom'
        elif 'bed' in layer_name or 'chambre' in layer_name:
            return 'Bedroom'
        elif 'living' in layer_name or 'salon' in layer_name:
            return 'Living Room'
        else:
            return 'Room'

    def _calculate_polygon_area(self, points) -> float:
        """Calculate area of polygon from ezdxf points"""
        try:
            coords = [(float(p.x), float(p.y)) for p in points]
            return self._calculate_polygon_area_coords(coords)
        except:
            return 0.0

    def _calculate_polygon_area_coords(self, coords: List[tuple]) -> float:
        """Calculate area using shoelace formula"""
        if len(coords) < 3:
            return 0.0

        area = 0.0
        n = len(coords)
        for i in range(n):
            j = (i + 1) % n
            area += coords[i][0] * coords[j][1]
            area -= coords[j][0] * coords[i][1]
        return abs(area) / 2.0

    def _create_circle_points(self, center, radius, num_points=16) -> List[tuple]:
        """Create polygon points approximating a circle"""
        import math
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            points.append((float(x), float(y)))
        return points

    def _connect_lines_to_polygons(self, lines) -> List[List[tuple]]:
        """Connect LINE entities into closed polygons"""
        polygons = []

        try:
            # Convert lines to coordinate pairs
            line_segments = []
            for line in lines:
                start = (float(line.dxf.start.x), float(line.dxf.start.y))
                end = (float(line.dxf.end.x), float(line.dxf.end.y))
                line_segments.append((start, end))

            # Simple polygon detection (connect lines that share endpoints)
            used_segments = set()

            for i, segment in enumerate(line_segments):
                if i in used_segments:
                    continue

                # Try to build a polygon starting from this segment
                polygon = [segment[0], segment[1]]
                current_end = segment[1]
                used_segments.add(i)

                # Find connecting segments
                max_connections = 20  # Prevent infinite loops
                connections = 0

                while connections < max_connections:
                    found_connection = False

                    for j, other_segment in enumerate(line_segments):
                        if j in used_segments:
                            continue

                        # Check if segments connect
                        if self._points_close(current_end, other_segment[0], tolerance=0.1):
                            polygon.append(other_segment[1])
                            current_end = other_segment[1]
                            used_segments.add(j)
                            found_connection = True
                            break
                        elif self._points_close(current_end, other_segment[1], tolerance=0.1):
                            polygon.append(other_segment[0])
                            current_end = other_segment[0]
                            used_segments.add(j)
                            found_connection = True
                            break

                    if not found_connection:
                        break

                    # Check if we've closed the polygon
                    if self._points_close(current_end, polygon[0], tolerance=0.1):
                        polygon.pop()  # Remove duplicate closing point
                        break

                    connections += 1

                # Only add if we have a valid polygon
                if len(polygon) >= 3:
                    polygons.append(polygon)

            return polygons

        except Exception as e:
            print(f"Warning: Error connecting lines to polygons: {str(e)}")
            return []

    def _points_close(self, p1: tuple, p2: tuple, tolerance: float = 0.1) -> bool:
        """Check if two points are close enough to be considered connected"""
        import math
        distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        return distance <= tolerance