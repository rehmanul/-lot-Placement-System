<<<<<<< HEAD

"""
Enhanced DWG/DXF Parser - Enterprise Grade
Supports ALL CAD file formats with advanced recovery
=======
"""
Real DWG/DXF Parser - No Fallback Version
Only processes actual DWG/DXF files with real geometric data
>>>>>>> origin/replit-agent
"""

import tempfile
import os
<<<<<<< HEAD
import math
from typing import List, Dict, Any, Optional, Tuple
=======
from typing import List, Dict, Any, Optional
>>>>>>> origin/replit-agent
import traceback

try:
    import ezdxf
<<<<<<< HEAD
    from ezdxf import recover
=======
>>>>>>> origin/replit-agent
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False

<<<<<<< HEAD
try:
    import dxfgrabber
    DXFGRABBER_AVAILABLE = True
except ImportError:
    DXFGRABBER_AVAILABLE = False

class DWGParser:
    def __init__(self):
        self.supported_formats = ['.dwg', '.dxf']
        self.entity_types = [
            'LWPOLYLINE', 'POLYLINE', 'LINE', 'CIRCLE', 'ARC', 
            'ELLIPSE', 'SPLINE', 'HATCH', 'SOLID', 'TRACE',
            'INSERT', 'TEXT', 'MTEXT', 'DIMENSION'
        ]

    def parse_file_simple(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Parse DWG/DXF file - ENTERPRISE VERSION with ALL format support
        """
        if not EZDXF_AVAILABLE:
            print("ERROR: ezdxf not available - cannot parse CAD files")
            return []

        file_ext = os.path.splitext(filename.lower())[1]
        if file_ext not in self.supported_formats:
            print(f"ERROR: File '{filename}' is not a supported CAD format")
            return []

        try:
=======
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
>>>>>>> origin/replit-agent
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
                temp_file.write(file_bytes)
                temp_file_path = temp_file.name

            try:
<<<<<<< HEAD
                zones = []
                
                if file_ext == '.dwg':
                    # Handle DWG files with multiple strategies
                    zones = self._parse_dwg_file(temp_file_path, filename)
                elif file_ext == '.dxf':
                    # Handle DXF files with multiple strategies
                    zones = self._parse_dxf_file(temp_file_path, filename)
=======
                # Parse real DWG/DXF file
                zones = self._parse_real_dwg_dxf(temp_file_path, filename)
>>>>>>> origin/replit-agent

                if not zones:
                    print(f"ERROR: No valid geometric data found in '{filename}'")
                    return []

                print(f"SUCCESS: Parsed {len(zones)} real zones from '{filename}'")
                return zones

            finally:
<<<<<<< HEAD
=======
                # Clean up temp file
>>>>>>> origin/replit-agent
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except Exception as e:
            print(f"ERROR: Failed to parse '{filename}': {str(e)}")
            traceback.print_exc()
            return []

<<<<<<< HEAD
    def _parse_dwg_file(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Parse DWG files with multiple recovery strategies"""
        doc = None
        
        # Strategy 1: Direct recovery mode (most compatible)
        try:
            doc, auditor = recover.readfile(file_path)
            if auditor.has_errors:
                print(f"WARNING: DWG file has {len(auditor.errors)} issues but was recovered")
            print(f"SUCCESS: DWG file opened with recovery mode: {filename}")
        except Exception as e:
            print(f"DWG recovery mode failed: {str(e)}")
        
        # Strategy 2: Try standard reading (fallback)
        if doc is None:
            try:
                doc = ezdxf.readfile(file_path)
                print(f"SUCCESS: DWG file opened with standard mode: {filename}")
            except Exception as e:
                print(f"DWG standard mode failed: {str(e)}")
        
        # Strategy 3: dxfgrabber fallback
        if doc is None and DXFGRABBER_AVAILABLE:
            try:
                return self._parse_with_dxfgrabber(file_path, filename)
            except Exception as e:
                print(f"DXFGrabber fallback failed: {str(e)}")
        
        if doc is None:
            print(f"ERROR: All DWG parsing strategies failed for {filename}")
            return []
        
        return self._extract_zones_from_doc(doc, filename)

    def _parse_dxf_file(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Parse DXF files with multiple strategies"""
        doc = None
        
        # Strategy 1: Standard reading
        try:
            doc = ezdxf.readfile(file_path)
            print(f"SUCCESS: DXF file opened with standard mode: {filename}")
        except Exception as e:
            print(f"DXF standard mode failed: {str(e)}")
        
        # Strategy 2: Recovery mode
        if doc is None:
            try:
                doc, auditor = recover.readfile(file_path)
                if auditor.has_errors:
                    print(f"WARNING: DXF file has {len(auditor.errors)} issues but was recovered")
                print(f"SUCCESS: DXF file opened with recovery mode: {filename}")
            except Exception as e:
                print(f"DXF recovery mode failed: {str(e)}")
        
        # Strategy 3: dxfgrabber fallback
        if doc is None and DXFGRABBER_AVAILABLE:
            try:
                return self._parse_with_dxfgrabber(file_path, filename)
            except Exception as e:
                print(f"DXFGrabber fallback failed: {str(e)}")
        
        if doc is None:
            print(f"ERROR: All DXF parsing strategies failed for {filename}")
            return []
        
        return self._extract_zones_from_doc(doc, filename)

    def _parse_with_dxfgrabber(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Fallback parsing using dxfgrabber library"""
        try:
            dxf = dxfgrabber.readfile(file_path)
            zones = []
            entities_processed = 0
            
            # Process different entity types
            for entity in dxf.entities:
                try:
                    if entity.dxftype == 'LWPOLYLINE':
                        zone = self._process_dxfgrabber_lwpolyline(entity, entities_processed)
                        if zone:
                            zones.append(zone)
                            entities_processed += 1
                    elif entity.dxftype == 'POLYLINE':
                        zone = self._process_dxfgrabber_polyline(entity, entities_processed)
                        if zone:
                            zones.append(zone)
                            entities_processed += 1
                    elif entity.dxftype == 'CIRCLE':
                        zone = self._process_dxfgrabber_circle(entity, entities_processed)
                        if zone:
                            zones.append(zone)
                            entities_processed += 1
                except Exception as e:
                    print(f"Warning: Error processing entity {entity.dxftype}: {str(e)}")
                    continue
            
            print(f"DXFGrabber parsed {entities_processed} entities from {filename}")
            return zones
            
        except Exception as e:
            print(f"DXFGrabber parsing failed: {str(e)}")
            return []

    def _extract_zones_from_doc(self, doc, filename: str) -> List[Dict[str, Any]]:
        """Extract zones from ezdxf document"""
        zones = []
        msp = doc.modelspace()
        entities_processed = 0

        # Process all supported entity types
        for entity_type in self.entity_types:
            try:
                entities = list(msp.query(entity_type))
                if entities:
                    print(f"Processing {len(entities)} {entity_type} entities")
                    
                    for entity in entities:
                        try:
                            zone = self._process_entity(entity, entity_type, entities_processed)
                            if zone:
                                zones.append(zone)
                                entities_processed += 1
                        except Exception as e:
                            print(f"Warning: Error processing {entity_type}: {str(e)}")
                            continue
                            
            except Exception as e:
                print(f"Warning: Error querying {entity_type}: {str(e)}")
                continue

        # Try to connect LINE entities into polygons
        try:
=======
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
>>>>>>> origin/replit-agent
            lines = list(msp.query('LINE'))
            if lines:
                connected_polygons = self._connect_lines_to_polygons(lines)
                for i, polygon in enumerate(connected_polygons):
                    zone = {
<<<<<<< HEAD
                        'zone_id': f"line_polygon_{entities_processed + i}",
=======
                        'zone_id': f"line_polygon_{entities_processed}",
>>>>>>> origin/replit-agent
                        'zone_type': 'Room',
                        'points': polygon,
                        'area': self._calculate_polygon_area_coords(polygon),
                        'layer': 'LINE_ASSEMBLY',
                        'entity_type': 'LINE_POLYGON'
                    }
                    zones.append(zone)
                    entities_processed += 1
<<<<<<< HEAD
        except Exception as e:
            print(f"Warning: Error connecting lines: {str(e)}")

        print(f"Total entities processed: {entities_processed}, zones created: {len(zones)}")
        return zones

    def _process_entity(self, entity, entity_type: str, index: int) -> Optional[Dict[str, Any]]:
        """Process individual entity based on type"""
        try:
            if entity_type == 'LWPOLYLINE':
                return self._process_lwpolyline(entity, index)
            elif entity_type == 'POLYLINE':
                return self._process_polyline(entity, index)
            elif entity_type == 'CIRCLE':
                return self._process_circle(entity, index)
            elif entity_type == 'ARC':
                return self._process_arc(entity, index)
            elif entity_type == 'ELLIPSE':
                return self._process_ellipse(entity, index)
            elif entity_type == 'SPLINE':
                return self._process_spline(entity, index)
            elif entity_type == 'HATCH':
                return self._process_hatch(entity, index)
            elif entity_type == 'INSERT':
                return self._process_insert(entity, index)
            else:
                return None
        except Exception as e:
            print(f"Error processing {entity_type}: {str(e)}")
            return None

    def _process_lwpolyline(self, entity, index: int) -> Optional[Dict[str, Any]]:
        """Process LWPOLYLINE entity"""
        try:
            points = list(entity.vertices_in_wcs())
            if len(points) >= 3:
                return {
                    'zone_id': f"lwpoly_{index}",
                    'zone_type': self._classify_entity_type(entity),
                    'points': [(float(p.x), float(p.y)) for p in points],
                    'area': self._calculate_polygon_area(points),
                    'layer': getattr(entity.dxf, 'layer', 'Unknown'),
                    'entity_type': 'LWPOLYLINE',
                    'is_closed': entity.closed
                }
        except Exception as e:
            print(f"Error processing LWPOLYLINE: {str(e)}")
        return None

    def _process_polyline(self, entity, index: int) -> Optional[Dict[str, Any]]:
        """Process POLYLINE entity"""
        try:
            points = list(entity.vertices_in_wcs())
            if len(points) >= 3:
                return {
                    'zone_id': f"poly_{index}",
                    'zone_type': self._classify_entity_type(entity),
                    'points': [(float(p.x), float(p.y)) for p in points],
                    'area': self._calculate_polygon_area(points),
                    'layer': getattr(entity.dxf, 'layer', 'Unknown'),
                    'entity_type': 'POLYLINE',
                    'is_closed': entity.is_closed
                }
        except Exception as e:
            print(f"Error processing POLYLINE: {str(e)}")
        return None

    def _process_circle(self, entity, index: int) -> Optional[Dict[str, Any]]:
        """Process CIRCLE entity"""
        try:
            center = entity.dxf.center
            radius = entity.dxf.radius
            points = self._create_circle_points(center, radius)
            
            return {
                'zone_id': f"circle_{index}",
                'zone_type': self._classify_entity_type(entity),
                'points': points,
                'area': math.pi * radius * radius,
                'layer': getattr(entity.dxf, 'layer', 'Unknown'),
                'entity_type': 'CIRCLE',
                'center': (float(center.x), float(center.y)),
                'radius': float(radius)
            }
        except Exception as e:
            print(f"Error processing CIRCLE: {str(e)}")
        return None

    def _process_arc(self, entity, index: int) -> Optional[Dict[str, Any]]:
        """Process ARC entity"""
        try:
            center = entity.dxf.center
            radius = entity.dxf.radius
            start_angle = entity.dxf.start_angle
            end_angle = entity.dxf.end_angle
            
            points = self._create_arc_points(center, radius, start_angle, end_angle)
            
            return {
                'zone_id': f"arc_{index}",
                'zone_type': self._classify_entity_type(entity),
                'points': points,
                'area': 0,  # Arcs don't have area
                'layer': getattr(entity.dxf, 'layer', 'Unknown'),
                'entity_type': 'ARC',
                'center': (float(center.x), float(center.y)),
                'radius': float(radius),
                'start_angle': float(start_angle),
                'end_angle': float(end_angle)
            }
        except Exception as e:
            print(f"Error processing ARC: {str(e)}")
        return None

    def _process_ellipse(self, entity, index: int) -> Optional[Dict[str, Any]]:
        """Process ELLIPSE entity"""
        try:
            center = entity.dxf.center
            major_axis = entity.dxf.major_axis
            ratio = entity.dxf.ratio
            
            points = self._create_ellipse_points(center, major_axis, ratio)
            
            return {
                'zone_id': f"ellipse_{index}",
                'zone_type': self._classify_entity_type(entity),
                'points': points,
                'area': math.pi * abs(major_axis.x) * abs(major_axis.y) * ratio,
                'layer': getattr(entity.dxf, 'layer', 'Unknown'),
                'entity_type': 'ELLIPSE'
            }
        except Exception as e:
            print(f"Error processing ELLIPSE: {str(e)}")
        return None

    def _process_spline(self, entity, index: int) -> Optional[Dict[str, Any]]:
        """Process SPLINE entity"""
        try:
            points = self._spline_to_points(entity)
            if len(points) >= 2:
                return {
                    'zone_id': f"spline_{index}",
                    'zone_type': self._classify_entity_type(entity),
                    'points': points,
                    'area': self._calculate_polygon_area_coords(points) if len(points) >= 3 else 0,
                    'layer': getattr(entity.dxf, 'layer', 'Unknown'),
                    'entity_type': 'SPLINE',
                    'is_closed': entity.closed
                }
        except Exception as e:
            print(f"Error processing SPLINE: {str(e)}")
        return None

    def _process_hatch(self, entity, index: int) -> Optional[Dict[str, Any]]:
        """Process HATCH entity"""
        try:
            # Extract boundary paths from hatch
            all_points = []
            
            for path in entity.paths:
                if hasattr(path, 'edges'):
                    for edge in path.edges:
                        if hasattr(edge, 'start') and hasattr(edge, 'end'):
                            all_points.extend([
                                (float(edge.start.x), float(edge.start.y)),
                                (float(edge.end.x), float(edge.end.y))
                            ])
                elif hasattr(path, 'vertices'):
                    all_points.extend([
                        (float(v.x), float(v.y)) for v in path.vertices
                    ])
            
            if len(all_points) >= 3:
                return {
                    'zone_id': f"hatch_{index}",
                    'zone_type': self._classify_entity_type(entity),
                    'points': all_points,
                    'area': self._calculate_polygon_area_coords(all_points),
                    'layer': getattr(entity.dxf, 'layer', 'Unknown'),
                    'entity_type': 'HATCH'
                }
        except Exception as e:
            print(f"Error processing HATCH: {str(e)}")
        return None

    def _process_insert(self, entity, index: int) -> Optional[Dict[str, Any]]:
        """Process INSERT (block reference) entity"""
        try:
            # Get block definition and transform points
            block = entity.block()
            if block is None:
                return None
            
            insert_point = entity.dxf.insert
            scale_x = getattr(entity.dxf, 'xscale', 1.0)
            scale_y = getattr(entity.dxf, 'yscale', 1.0)
            rotation = getattr(entity.dxf, 'rotation', 0.0)
            
            # Simple bounding box for the block
            width = scale_x * 2  # Approximate
            height = scale_y * 2  # Approximate
            
            points = [
                (insert_point.x - width/2, insert_point.y - height/2),
                (insert_point.x + width/2, insert_point.y - height/2),
                (insert_point.x + width/2, insert_point.y + height/2),
                (insert_point.x - width/2, insert_point.y + height/2)
            ]
            
            return {
                'zone_id': f"insert_{index}",
                'zone_type': 'Furniture',
                'points': points,
                'area': width * height,
                'layer': getattr(entity.dxf, 'layer', 'Unknown'),
                'entity_type': 'INSERT',
                'block_name': entity.dxf.name
            }
        except Exception as e:
            print(f"Error processing INSERT: {str(e)}")
        return None

    def _process_dxfgrabber_lwpolyline(self, entity, index: int) -> Optional[Dict[str, Any]]:
        """Process LWPOLYLINE using dxfgrabber"""
        try:
            points = [(float(p[0]), float(p[1])) for p in entity.points]
            if len(points) >= 3:
                return {
                    'zone_id': f"lwpoly_{index}",
                    'zone_type': 'Room',
                    'points': points,
                    'area': self._calculate_polygon_area_coords(points),
                    'layer': getattr(entity, 'layer', 'Unknown'),
                    'entity_type': 'LWPOLYLINE'
                }
        except Exception as e:
            print(f"Error processing dxfgrabber LWPOLYLINE: {str(e)}")
        return None

    def _process_dxfgrabber_polyline(self, entity, index: int) -> Optional[Dict[str, Any]]:
        """Process POLYLINE using dxfgrabber"""
        try:
            points = [(float(p[0]), float(p[1])) for p in entity.points]
            if len(points) >= 3:
                return {
                    'zone_id': f"poly_{index}",
                    'zone_type': 'Room',
                    'points': points,
                    'area': self._calculate_polygon_area_coords(points),
                    'layer': getattr(entity, 'layer', 'Unknown'),
                    'entity_type': 'POLYLINE'
                }
        except Exception as e:
            print(f"Error processing dxfgrabber POLYLINE: {str(e)}")
        return None

    def _process_dxfgrabber_circle(self, entity, index: int) -> Optional[Dict[str, Any]]:
        """Process CIRCLE using dxfgrabber"""
        try:
            center_x, center_y = entity.center[0], entity.center[1]
            radius = entity.radius
            points = self._create_circle_points_coords(center_x, center_y, radius)
            
            return {
                'zone_id': f"circle_{index}",
                'zone_type': 'Room',
                'points': points,
                'area': math.pi * radius * radius,
                'layer': getattr(entity, 'layer', 'Unknown'),
                'entity_type': 'CIRCLE'
            }
        except Exception as e:
            print(f"Error processing dxfgrabber CIRCLE: {str(e)}")
        return None

    def _classify_entity_type(self, entity) -> str:
        """Classify entity type based on layer name and properties"""
        try:
            layer_name = getattr(entity.dxf, 'layer', '').lower()
        except:
            layer_name = ''
=======

            print(f"Real parsing result: {entities_processed} entities processed, {len(zones)} zones created")
            return zones

        except Exception as e:
            print(f"ERROR: Real DWG/DXF parsing failed: {str(e)}")
            traceback.print_exc()
            return []

    def _classify_entity_type(self, entity) -> str:
        """Classify entity type based on layer name and properties"""
        layer_name = entity.dxf.layer.lower() if hasattr(entity.dxf, 'layer') else ''
>>>>>>> origin/replit-agent

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

<<<<<<< HEAD
    def _calculate_polygon_area_coords(self, coords: List[Tuple[float, float]]) -> float:
=======
    def _calculate_polygon_area_coords(self, coords: List[tuple]) -> float:
>>>>>>> origin/replit-agent
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

<<<<<<< HEAD
    def _create_circle_points(self, center, radius, num_points=32) -> List[Tuple[float, float]]:
        """Create polygon points approximating a circle"""
=======
    def _create_circle_points(self, center, radius, num_points=16) -> List[tuple]:
        """Create polygon points approximating a circle"""
        import math
>>>>>>> origin/replit-agent
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            points.append((float(x), float(y)))
        return points

<<<<<<< HEAD
    def _create_circle_points_coords(self, center_x: float, center_y: float, radius: float, num_points=32) -> List[Tuple[float, float]]:
        """Create circle points from coordinates"""
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append((float(x), float(y)))
        return points

    def _create_arc_points(self, center, radius, start_angle, end_angle, num_points=16) -> List[Tuple[float, float]]:
        """Create points for arc approximation"""
        points = []
        start_rad = math.radians(start_angle)
        end_rad = math.radians(end_angle)
        
        if end_rad < start_rad:
            end_rad += 2 * math.pi
        
        angle_step = (end_rad - start_rad) / num_points
        
        for i in range(num_points + 1):
            angle = start_rad + i * angle_step
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            points.append((float(x), float(y)))
        
        return points

    def _create_ellipse_points(self, center, major_axis, ratio, num_points=32) -> List[Tuple[float, float]]:
        """Create points for ellipse approximation"""
        points = []
        a = math.sqrt(major_axis.x**2 + major_axis.y**2)  # Semi-major axis
        b = a * ratio  # Semi-minor axis
        
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = center.x + a * math.cos(angle)
            y = center.y + b * math.sin(angle)
            points.append((float(x), float(y)))
        
        return points

    def _spline_to_points(self, spline, num_points=20) -> List[Tuple[float, float]]:
        """Convert spline to polyline approximation"""
        try:
            points = []
            for point in spline.flattening(0.1):
                points.append((float(point.x), float(point.y)))
            return points
        except:
            try:
                control_points = spline.control_points
                return [(float(p.x), float(p.y)) for p in control_points]
            except:
                return []

    def _connect_lines_to_polygons(self, lines) -> List[List[Tuple[float, float]]]:
        """Connect LINE entities into closed polygons"""
        polygons = []
        
        try:
=======
    def _connect_lines_to_polygons(self, lines) -> List[List[tuple]]:
        """Connect LINE entities into closed polygons"""
        polygons = []

        try:
            # Convert lines to coordinate pairs
>>>>>>> origin/replit-agent
            line_segments = []
            for line in lines:
                start = (float(line.dxf.start.x), float(line.dxf.start.y))
                end = (float(line.dxf.end.x), float(line.dxf.end.y))
                line_segments.append((start, end))
<<<<<<< HEAD
            
            used_segments = set()
            
            for i, segment in enumerate(line_segments):
                if i in used_segments:
                    continue
                
                polygon = [segment[0], segment[1]]
                current_end = segment[1]
                used_segments.add(i)
                
                max_connections = 20
                connections = 0
                
                while connections < max_connections:
                    found_connection = False
                    
                    for j, other_segment in enumerate(line_segments):
                        if j in used_segments:
                            continue
                        
=======

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
>>>>>>> origin/replit-agent
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
<<<<<<< HEAD
                    
                    if not found_connection:
                        break
                    
                    if self._points_close(current_end, polygon[0], tolerance=0.1):
                        polygon.pop()
                        break
                    
                    connections += 1
                
                if len(polygon) >= 3:
                    polygons.append(polygon)
            
            return polygons
            
=======

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

>>>>>>> origin/replit-agent
        except Exception as e:
            print(f"Warning: Error connecting lines to polygons: {str(e)}")
            return []

<<<<<<< HEAD
    def _points_close(self, p1: Tuple[float, float], p2: Tuple[float, float], tolerance: float = 0.1) -> bool:
        """Check if two points are close enough to be considered connected"""
        distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        return distance <= tolerance
=======
    def _points_close(self, p1: tuple, p2: tuple, tolerance: float = 0.1) -> bool:
        """Check if two points are close enough to be considered connected"""
        import math
        distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        return distance <= tolerance
>>>>>>> origin/replit-agent
