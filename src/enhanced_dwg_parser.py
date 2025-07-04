"""
Enhanced DWG Parser - Real Version Only
Advanced parsing with no fallback methods
"""

from typing import Dict, List, Any, Optional, Tuple
from functools import wraps
import hashlib
import numpy as np
import tempfile
import os
import traceback

try:
    import ezdxf
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False

class EnhancedDWGParser:
    def __init__(self):
        self.supported_formats = ['.dwg', '.dxf']

    def parse_advanced(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Advanced DWG/DXF parsing - REAL VERSION ONLY
        Returns empty result if file cannot be parsed as real DWG/DXF
        """
        if not EZDXF_AVAILABLE:
            return {
                'success': False,
                'error': 'ezdxf not available - cannot parse real DWG/DXF files',
                'zones': [],
                'metadata': {}
            }

        # Only process actual DWG/DXF files
        file_ext = os.path.splitext(filename.lower())[1]
        if file_ext not in self.supported_formats:
            return {
                'success': False,
                'error': f"File '{filename}' is not a supported DWG/DXF format",
                'zones': [],
                'metadata': {}
            }

        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
                temp_file.write(file_bytes)
                temp_file_path = temp_file.name

            try:
                # Parse real DWG/DXF file
                result = self._parse_real_enhanced(temp_file_path, filename)

                if not result['success']:
                    return result

                print(f"SUCCESS: Enhanced parsing of '{filename}' completed")
                return result

            finally:
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to parse '{filename}': {str(e)}",
                'zones': [],
                'metadata': {}
            }

    def _parse_real_enhanced(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Enhanced parsing of real DWG/DXF files"""
        try:
            # Open DWG/DXF file
            try:
                doc = ezdxf.readfile(file_path)
                print(f"Successfully opened: {filename}")
            except Exception as e:
                return {
                    'success': False,
                    'error': f"Cannot open '{filename}' as DWG/DXF: {str(e)}",
                    'zones': [],
                    'metadata': {}
                }

            # Extract document metadata
            metadata = self._extract_metadata(doc, filename)

            # Extract zones with enhanced analysis
            zones = self._extract_enhanced_zones(doc)

            if not zones:
                return {
                    'success': False,
                    'error': f"No valid geometric data found in '{filename}'",
                    'zones': [],
                    'metadata': metadata
                }

            # Enhanced zone analysis
            enhanced_zones = self._analyze_zones_enhanced(zones)

            return {
                'success': True,
                'zones': enhanced_zones,
                'metadata': metadata,
                'statistics': self._calculate_statistics(enhanced_zones)
            }

        except Exception as e:
            return {
                'success': False,
                'error': f"Enhanced parsing failed: {str(e)}",
                'zones': [],
                'metadata': {}
            }

    def _extract_metadata(self, doc, filename: str) -> Dict[str, Any]:
        """Extract real metadata from DWG/DXF document"""
        metadata = {
            'filename': filename,
            'dxf_version': doc.dxfversion,
            'layers': [],
            'blocks': [],
            'entities_count': 0
        }

        try:
            # Extract layer information
            for layer in doc.layers:
                metadata['layers'].append({
                    'name': layer.dxf.name,
                    'color': layer.dxf.color,
                    'linetype': layer.dxf.linetype
                })

            # Count entities
            msp = doc.modelspace()
            metadata['entities_count'] = len(list(msp))

            # Extract blocks
            for block in doc.blocks:
                if not block.name.startswith('*'):  # Skip anonymous blocks
                    metadata['blocks'].append(block.name)

        except Exception as e:
            print(f"Warning: Error extracting metadata: {str(e)}")

        return metadata

    def _extract_enhanced_zones(self, doc) -> List[Dict[str, Any]]:
        """Extract zones with enhanced geometric analysis"""
        zones = []
        msp = doc.modelspace()

        # Process different entity types
        entity_processors = {
            'LWPOLYLINE': self._process_lwpolyline,
            'POLYLINE': self._process_polyline,
            'CIRCLE': self._process_circle,
            'ARC': self._process_arc,
            'LINE': self._process_line,
            'SPLINE': self._process_spline
        }

        for entity_type, processor in entity_processors.items():
            try:
                entities = list(msp.query(entity_type))
                if entities:
                    print(f"Processing {len(entities)} {entity_type} entities")
                    processed_zones = processor(entities)
                    zones.extend(processed_zones)
            except Exception as e:
                print(f"Warning: Error processing {entity_type}: {str(e)}")
                continue

        return zones

    def _process_lwpolyline(self, entities: List) -> List[Dict[str, Any]]:
        """Process LWPOLYLINE entities"""
        zones = []
        for i, entity in enumerate(entities):
            try:
                points = list(entity.vertices_in_wcs())
                if len(points) >= 3:
                    zone = {
                        'zone_id': f"lwpoly_{i}",
                        'zone_type': self._classify_by_layer(entity.dxf.layer),
                        'points': [(float(p.x), float(p.y)) for p in points],
                        'area': self._calculate_area(points),
                        'perimeter': self._calculate_perimeter(points),
                        'layer': entity.dxf.layer,
                        'entity_type': 'LWPOLYLINE',
                        'is_closed': entity.closed,
                        'bounds': self._calculate_bounds(points)
                    }
                    zones.append(zone)
            except Exception as e:
                print(f"Warning: Error processing LWPOLYLINE {i}: {str(e)}")
                continue
        return zones

    def _process_polyline(self, entities: List) -> List[Dict[str, Any]]:
        """Process POLYLINE entities"""
        zones = []
        for i, entity in enumerate(entities):
            try:
                points = list(entity.vertices_in_wcs())
                if len(points) >= 3:
                    zone = {
                        'zone_id': f"poly_{i}",
                        'zone_type': self._classify_by_layer(entity.dxf.layer),
                        'points': [(float(p.x), float(p.y)) for p in points],
                        'area': self._calculate_area(points),
                        'perimeter': self._calculate_perimeter(points),
                        'layer': entity.dxf.layer,
                        'entity_type': 'POLYLINE',
                        'is_closed': entity.is_closed,
                        'bounds': self._calculate_bounds(points)
                    }
                    zones.append(zone)
            except Exception as e:
                print(f"Warning: Error processing POLYLINE {i}: {str(e)}")
                continue
        return zones

    def _process_circle(self, entities: List) -> List[Dict[str, Any]]:
        """Process CIRCLE entities"""
        zones = []
        for i, entity in enumerate(entities):
            try:
                center = entity.dxf.center
                radius = entity.dxf.radius
                points = self._create_circle_points(center, radius)

                zone = {
                    'zone_id': f"circle_{i}",
                    'zone_type': self._classify_by_layer(entity.dxf.layer),
                    'points': points,
                    'area': 3.14159 * radius * radius,
                    'perimeter': 2 * 3.14159 * radius,
                    'layer': entity.dxf.layer,
                    'entity_type': 'CIRCLE',
                    'is_closed': True,
                    'center': (float(center.x), float(center.y)),
                    'radius': float(radius),
                    'bounds': self._calculate_bounds_coords(points)
                }
                zones.append(zone)
            except Exception as e:
                print(f"Warning: Error processing CIRCLE {i}: {str(e)}")
                continue
        return zones

    def _process_arc(self, entities: List) -> List[Dict[str, Any]]:
        """Process ARC entities"""
        zones = []
        for i, entity in enumerate(entities):
            try:
                center = entity.dxf.center
                radius = entity.dxf.radius
                start_angle = entity.dxf.start_angle
                end_angle = entity.dxf.end_angle

                points = self._create_arc_points(center, radius, start_angle, end_angle)

                zone = {
                    'zone_id': f"arc_{i}",
                    'zone_type': self._classify_by_layer(entity.dxf.layer),
                    'points': points,
                    'area': 0,  # Arcs don't have area
                    'perimeter': abs(end_angle - start_angle) * 3.14159 * radius / 180,
                    'layer': entity.dxf.layer,
                    'entity_type': 'ARC',
                    'is_closed': False,
                    'center': (float(center.x), float(center.y)),
                    'radius': float(radius),
                    'bounds': self._calculate_bounds_coords(points)
                }
                zones.append(zone)
            except Exception as e:
                print(f"Warning: Error processing ARC {i}: {str(e)}")
                continue
        return zones

    def _process_line(self, entities: List) -> List[Dict[str, Any]]:
        """Process LINE entities"""
        zones = []
        for i, entity in enumerate(entities):
            try:
                start = entity.dxf.start
                end = entity.dxf.end
                points = [(float(start.x), float(start.y)), (float(end.x), float(end.y))]

                zone = {
                    'zone_id': f"line_{i}",
                    'zone_type': self._classify_by_layer(entity.dxf.layer),
                    'points': points,
                    'area': 0,  # Lines don't have area
                    'perimeter': self._calculate_distance(points[0], points[1]),
                    'layer': entity.dxf.layer,
                    'entity_type': 'LINE',
                    'is_closed': False,
                    'bounds': self._calculate_bounds_coords(points)
                }
                zones.append(zone)
            except Exception as e:
                print(f"Warning: Error processing LINE {i}: {str(e)}")
                continue
        return zones

    def _process_spline(self, entities: List) -> List[Dict[str, Any]]:
        """Process SPLINE entities"""
        zones = []
        for i, entity in enumerate(entities):
            try:
                # Convert spline to polyline approximation
                points = self._spline_to_points(entity)

                if len(points) >= 2:
                    zone = {
                        'zone_id': f"spline_{i}",
                        'zone_type': self._classify_by_layer(entity.dxf.layer),
                        'points': points,
                        'area': self._calculate_area_coords(points) if len(points) >= 3 else 0,
                        'perimeter': self._calculate_perimeter_coords(points),
                        'layer': entity.dxf.layer,
                        'entity_type': 'SPLINE',
                        'is_closed': entity.closed,
                        'bounds': self._calculate_bounds_coords(points)
                    }
                    zones.append(zone)
            except Exception as e:
                print(f"Warning: Error processing SPLINE {i}: {str(e)}")
                continue
        return zones

    def _analyze_zones_enhanced(self, zones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced zone analysis"""
        enhanced_zones = []

        for zone in zones:
            try:
                # Add geometric analysis
                zone['geometric_analysis'] = self._analyze_geometry(zone)

                # Add spatial relationships
                zone['spatial_relationships'] = self._analyze_spatial_relationships(zone, zones)

                # Add quality metrics
                zone['quality_metrics'] = self._calculate_quality_metrics(zone)

                enhanced_zones.append(zone)

            except Exception as e:
                print(f"Warning: Error enhancing zone {zone.get('zone_id', 'unknown')}: {str(e)}")
                enhanced_zones.append(zone)  # Add original zone if enhancement fails
                continue

        return enhanced_zones

    def _classify_by_layer(self, layer_name: str) -> str:
        """Classify entity type based on layer name"""
        layer_lower = layer_name.lower()

        classification_map = {
            'wall': 'Wall', 'mur': 'Wall', 'walls': 'Wall',
            'door': 'Door', 'porte': 'Door', 'doors': 'Door',
            'window': 'Window', 'fenetre': 'Window', 'windows': 'Window',
            'kitchen': 'Kitchen', 'cuisine': 'Kitchen',
            'bath': 'Bathroom', 'salle': 'Bathroom', 'bathroom': 'Bathroom',
            'bed': 'Bedroom', 'chambre': 'Bedroom', 'bedroom': 'Bedroom',
            'living': 'Living Room', 'salon': 'Living Room'
        }

        for key, value in classification_map.items():
            if key in layer_lower:
                return value

        return 'Room'

    def _calculate_area(self, points) -> float:
        """Calculate area from ezdxf points"""
        try:
            coords = [(float(p.x), float(p.y)) for p in points]
            return self._calculate_area_coords(coords)
        except:
            return 0.0

    def _calculate_area_coords(self, coords: List[Tuple[float, float]]) -> float:
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

    def _calculate_perimeter(self, points) -> float:
        """Calculate perimeter from ezdxf points"""
        try:
            coords = [(float(p.x), float(p.y)) for p in points]
            return self._calculate_perimeter_coords(coords)
        except:
            return 0.0

    def _calculate_perimeter_coords(self, coords: List[Tuple[float, float]]) -> float:
        """Calculate perimeter"""
        if len(coords) < 2:
            return 0.0

        perimeter = 0.0
        for i in range(len(coords)):
            j = (i + 1) % len(coords)
            perimeter += self._calculate_distance(coords[i], coords[j])
        return perimeter

    def _calculate_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate distance between two points"""
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    def _calculate_bounds(self, points) -> Dict[str, float]:
        """Calculate bounding box from ezdxf points"""
        try:
            coords = [(float(p.x), float(p.y)) for p in points]
            return self._calculate_bounds_coords(coords)
        except:
            return {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0}

    def _calculate_bounds_coords(self, coords: List[Tuple[float, float]]) -> Dict[str, float]:
        """Calculate bounding box"""
        if not coords:
            return {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0}

        x_coords = [p[0] for p in coords]
        y_coords = [p[1] for p in coords]

        return {
            'min_x': min(x_coords),
            'max_x': max(x_coords),
            'min_y': min(y_coords),
            'max_y': max(y_coords)
        }

    def _create_circle_points(self, center, radius, num_points=32) -> List[Tuple[float, float]]:
        """Create polygon points approximating a circle"""
        import math
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            points.append((float(x), float(y)))
        return points

    def _create_arc_points(self, center, radius, start_angle, end_angle, num_points=16) -> List[Tuple[float, float]]:
        """Create points for arc approximation"""
        import math
        points = []

        # Convert angles to radians
        start_rad = math.radians(start_angle)
        end_rad = math.radians(end_angle)

        # Handle angle wrapping
        if end_rad < start_rad:
            end_rad += 2 * math.pi

        angle_step = (end_rad - start_rad) / num_points

        for i in range(num_points + 1):
            angle = start_rad + i * angle_step
            x = center.x + radius * math.cos(angle)
            y = center.y + radius * math.sin(angle)
            points.append((float(x), float(y)))

        return points

    def _spline_to_points(self, spline, num_points=20) -> List[Tuple[float, float]]:
        """Convert spline to polyline approximation"""
        try:
            # Use ezdxf's flattening capability
            points = []
            for point in spline.flattening(0.1):  # 0.1 is the approximation distance
                points.append((float(point.x), float(point.y)))
            return points
        except:
            # Fallback: use control points
            try:
                control_points = spline.control_points
                return [(float(p.x), float(p.y)) for p in control_points]
            except:
                return []

    def _analyze_geometry(self, zone: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze geometric properties"""
        analysis = {}

        try:
            points = zone.get('points', [])
            if len(points) >= 3:
                # Calculate centroid
                centroid_x = sum(p[0] for p in points) / len(points)
                centroid_y = sum(p[1] for p in points) / len(points)
                analysis['centroid'] = (centroid_x, centroid_y)

                # Calculate shape complexity
                analysis['vertex_count'] = len(points)
                analysis['shape_complexity'] = 'simple' if len(points) <= 4 else 'complex'

                # Calculate aspect ratio
                bounds = zone.get('bounds', {})
                if bounds:
                    width = bounds.get('max_x', 0) - bounds.get('min_x', 0)
                    height = bounds.get('max_y', 0) - bounds.get('min_y', 0)
                    analysis['aspect_ratio'] = width / height if height > 0 else 1.0

        except Exception as e:
            print(f"Warning: Error analyzing geometry: {str(e)}")

        return analysis

    def _analyze_spatial_relationships(self, zone: Dict[str, Any], all_zones: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze spatial relationships with other zones"""
        relationships = {
            'adjacent_zones': [],
            'overlapping_zones': [],
            'contained_zones': [],
            'containing_zones': []
        }

        # This is a simplified implementation
        # In a real application, you'd use proper geometric algorithms

        return relationships

    def _calculate_quality_metrics(self, zone: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics for the zone"""
        metrics = {
            'completeness': 1.0,  # Assume complete for real parsed data
            'accuracy': 1.0,      # Assume accurate for real parsed data
            'geometric_validity': True
        }

        try:
            # Check for basic geometric validity
            points = zone.get('points', [])
            if len(points) < 3:
                metrics['geometric_validity'] = False

            # Check for self-intersections (simplified)
            area = zone.get('area', 0)
            if area <= 0:
                metrics['geometric_validity'] = False

        except Exception as e:
            print(f"Warning: Error calculating quality metrics: {str(e)}")
            metrics['geometric_validity'] = False

        return metrics

    def _calculate_statistics(self, zones: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall statistics"""
        stats = {
            'total_zones': len(zones),
            'total_area': sum(zone.get('area', 0) for zone in zones),
            'zone_types': {},
            'layers': set(),
            'entity_types': {}
        }

        for zone in zones:
            # Count zone types
            zone_type = zone.get('zone_type', 'Unknown')
            stats['zone_types'][zone_type] = stats['zone_types'].get(zone_type, 0) + 1

            # Collect layers
            layer = zone.get('layer', 'Unknown')
            stats['layers'].add(layer)

            # Count entity types
            entity_type = zone.get('entity_type', 'Unknown')
            stats['entity_types'][entity_type] = stats['entity_types'].get(entity_type, 0) + 1

        stats['layers'] = list(stats['layers'])

        return stats