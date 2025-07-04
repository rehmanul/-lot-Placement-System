
"""
Enterprise-Grade Îlot Placement Engine
Specialized for îlot placement with size distributions and corridor generation
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from shapely.geometry import Polygon, Point, box, LineString
from shapely.ops import unary_union
import random
import math


class IlotPlacementEngine:
    """Enterprise-grade îlot placement engine with size distribution and corridor generation"""
    
    def __init__(self):
        self.zone_types = {
            'AVAILABLE': {'color': 'white', 'placeable': True},
            'NO_ENTREE': {'color': 'lightblue', 'placeable': False},  # Stairs, elevators
            'ENTREE_SORTIE': {'color': 'red', 'placeable': False},    # Entrances/exits
            'MUR': {'color': 'black', 'placeable': True, 'wall': True}  # Walls
        }
        
        self.corridor_width = 1.2  # Default corridor width in meters
        self.min_distance_from_entrance = 2.0  # Minimum distance from entrances
        
    def place_ilots_with_distribution(self, zones: List[Dict], distribution: Dict[str, float], 
                                    total_area: float, corridor_width: float = 1.2) -> Dict[str, Any]:
        """
        Place îlots according to size distribution requirements
        
        Args:
            zones: List of detected zones with type classification
            distribution: Size distribution (e.g., {'0-1': 10, '1-3': 25, '3-5': 30, '5-10': 35})
            total_area: Total available area for îlot placement
            corridor_width: Width of corridors between îlot rows
            
        Returns:
            Placement results with îlots and corridors
        """
        self.corridor_width = corridor_width
        
        # Filter available zones (exclude entrances and restricted areas)
        available_zones = self._filter_available_zones(zones)
        
        if not available_zones:
            return {'error': 'No available zones for îlot placement'}
        
        # Calculate îlot counts based on distribution
        ilot_counts = self._calculate_ilot_counts(distribution, total_area)
        
        # Generate îlots with specified sizes
        ilots_to_place = self._generate_ilots_by_size(ilot_counts)
        
        # Place îlots in available zones
        placement_results = self._place_ilots_optimized(available_zones, ilots_to_place)
        
        # Generate corridors between facing îlot rows
        corridors = self._generate_corridors(placement_results['placed_ilots'], available_zones)
        
        # Validate placement constraints
        validation_results = self._validate_placement_constraints(
            placement_results['placed_ilots'], zones, corridors
        )
        
        return {
            'placed_ilots': placement_results['placed_ilots'],
            'corridors': corridors,
            'statistics': {
                'total_ilots': len(placement_results['placed_ilots']),
                'total_area_used': sum(ilot['area'] for ilot in placement_results['placed_ilots']),
                'distribution_achieved': self._calculate_achieved_distribution(placement_results['placed_ilots']),
                'space_efficiency': placement_results.get('efficiency', 0.0)
            },
            'validation': validation_results,
            'zones_used': available_zones
        }
    
    def _filter_available_zones(self, zones: List[Dict]) -> List[Dict]:
        """Filter zones that are available for îlot placement"""
        available_zones = []
        
        for zone in zones:
            zone_type = zone.get('zone_type', 'AVAILABLE').upper()
            
<<<<<<< HEAD
<<<<<<< HEAD
            # Only use zones that are explicitly available for placement
            if zone_type in ['AVAILABLE', 'MAIN_FLOOR', 'ROOM', 'SPACE']:
=======
=======
>>>>>>> origin/replit-agent
            # Accept zones that are suitable for placement (exclude only restricted areas)
            excluded_types = ['NO_ENTREE', 'ENTREE_SORTIE', 'MUR', 'WALL', 'ENTRANCE', 'EXIT', 'STAIRS', 'ELEVATOR']
            
            # If zone type is not explicitly excluded, consider it available
            if zone_type not in excluded_types:
<<<<<<< HEAD
>>>>>>> origin/replit-agent
=======
>>>>>>> origin/replit-agent
                # Ensure minimum area
                area = zone.get('area', 0)
                if area >= 1.0:  # Minimum 1m² for îlot placement
                    available_zones.append(zone)
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> origin/replit-agent
            # Also accept zones explicitly marked as available
            elif zone_type in ['AVAILABLE', 'MAIN_FLOOR', 'ROOM', 'SPACE', 'OFFICE', 'CHAMBER', 'SALON', 'LIVING']:
                area = zone.get('area', 0)
                if area >= 1.0:
                    available_zones.append(zone)
<<<<<<< HEAD
>>>>>>> origin/replit-agent
=======
>>>>>>> origin/replit-agent
        
        return available_zones
    
    def _calculate_ilot_counts(self, distribution: Dict[str, float], total_area: float) -> Dict[str, int]:
        """Calculate number of îlots for each size category"""
        ilot_counts = {}
        
        # Estimate average îlot density (îlots per m²)
        estimated_density = 0.15  # Adjustable based on requirements
        total_ilots = int(total_area * estimated_density)
        
        for size_range, percentage in distribution.items():
            count = int(total_ilots * (percentage / 100.0))
            ilot_counts[size_range] = max(1, count)  # Ensure at least 1 îlot per category
        
        return ilot_counts
    
    def _generate_ilots_by_size(self, ilot_counts: Dict[str, int]) -> List[Dict]:
        """Generate îlots with specified size ranges"""
        ilots = []
        
        size_ranges = {
            '0-1': (0.5, 1.0),
            '1-3': (1.0, 3.0),
            '3-5': (3.0, 5.0),
            '5-10': (5.0, 10.0)
        }
        
        for size_range, count in ilot_counts.items():
            if size_range in size_ranges:
                min_area, max_area = size_ranges[size_range]
                
                for i in range(count):
                    # Generate random area within range
                    area = random.uniform(min_area, max_area)
                    
                    # Calculate dimensions (assume roughly square with some variation)
                    base_side = math.sqrt(area)
                    aspect_ratio = random.uniform(0.7, 1.4)  # Variation in shape
                    
                    width = base_side * math.sqrt(aspect_ratio)
                    height = area / width
                    
                    ilots.append({
                        'id': f"{size_range}_{i+1}",
                        'size_range': size_range,
                        'area': area,
                        'width': width,
                        'height': height,
                        'priority': self._get_size_priority(size_range)
                    })
        
        # Sort by priority (larger îlots first for better placement)
        ilots.sort(key=lambda x: x['priority'], reverse=True)
        return ilots
    
    def _get_size_priority(self, size_range: str) -> int:
        """Get placement priority for size range (larger = higher priority)"""
        priorities = {'5-10': 4, '3-5': 3, '1-3': 2, '0-1': 1}
        return priorities.get(size_range, 1)
    
    def _place_ilots_optimized(self, available_zones: List[Dict], ilots: List[Dict]) -> Dict[str, Any]:
        """Optimized îlot placement algorithm"""
        placed_ilots = []
        failed_placements = []
        
        # Create combined placement area from all available zones
        zone_polygons = []
        for zone in available_zones:
            try:
                points = zone.get('points', [])
                if len(points) >= 3:
                    poly = Polygon(points)
                    if poly.is_valid and poly.area > 0:
                        zone_polygons.append(poly)
            except Exception:
                continue
        
        if not zone_polygons:
            return {'placed_ilots': [], 'efficiency': 0.0}
        
        # Combine all available areas
        try:
            combined_area = unary_union(zone_polygons)
        except Exception:
            combined_area = zone_polygons[0] if zone_polygons else None
        
        if not combined_area:
            return {'placed_ilots': [], 'efficiency': 0.0}
        
        # Grid-based placement with optimization
        for ilot in ilots:
            placement = self._find_optimal_placement(ilot, combined_area, placed_ilots)
            
            if placement:
                placed_ilots.append(placement)
            else:
                failed_placements.append(ilot)
        
        efficiency = len(placed_ilots) / len(ilots) if ilots else 0.0
        
        return {
            'placed_ilots': placed_ilots,
            'failed_placements': failed_placements,
            'efficiency': efficiency
        }
    
    def _find_optimal_placement(self, ilot: Dict, available_area: Polygon, 
                              existing_ilots: List[Dict]) -> Optional[Dict]:
        """Find optimal position for a single îlot"""
        width, height = ilot['width'], ilot['height']
        min_x, min_y, max_x, max_y = available_area.bounds
        
        # Grid search with optimization
        grid_spacing = min(width, height) / 2
        positions_tried = 0
        max_attempts = 500
        
        # Try multiple orientations
        orientations = [(width, height)]
        if abs(width - height) > 0.1:  # Only if significantly different
            orientations.append((height, width))
        
        for orientation in orientations:
            w, h = orientation
            
            # Grid search
            y = min_y + h/2
            while y + h/2 <= max_y and positions_tried < max_attempts:
                x = min_x + w/2
                while x + w/2 <= max_x and positions_tried < max_attempts:
                    positions_tried += 1
                    
                    # Create îlot rectangle
                    ilot_rect = box(x - w/2, y - h/2, x + w/2, y + h/2)
                    
                    # Check if within available area
                    if available_area.contains(ilot_rect):
                        # Check for overlaps with existing îlots
                        if not self._check_overlap(ilot_rect, existing_ilots):
                            return {
                                'id': ilot['id'],
                                'position': (x, y),
                                'width': w,
                                'height': h,
                                'area': ilot['area'],
                                'size_range': ilot['size_range'],
                                'bounds': [x - w/2, y - h/2, x + w/2, y + h/2],
                                'geometry': ilot_rect
                            }
                    
                    x += grid_spacing
                y += grid_spacing
        
        return None
    
    def _check_overlap(self, new_rect: Polygon, existing_ilots: List[Dict]) -> bool:
        """Check if new rectangle overlaps with existing îlots"""
        for ilot in existing_ilots:
            existing_rect = ilot.get('geometry')
            if existing_rect and new_rect.intersects(existing_rect):
                return True
        return False
    
    def _generate_corridors(self, placed_ilots: List[Dict], available_zones: List[Dict]) -> List[Dict]:
        """Generate corridors between facing îlot rows"""
        corridors = []
        
        if len(placed_ilots) < 2:
            return corridors
        
        # Group îlots by approximate rows (based on Y coordinates)
        rows = self._group_ilots_by_rows(placed_ilots)
        
        # Generate corridors between adjacent rows
        for i in range(len(rows) - 1):
            row1 = rows[i]
            row2 = rows[i + 1]
            
            corridor = self._create_corridor_between_rows(row1, row2)
            if corridor:
                corridors.append(corridor)
        
        return corridors
    
    def _group_ilots_by_rows(self, ilots: List[Dict]) -> List[List[Dict]]:
        """Group îlots into rows based on Y coordinates"""
        if not ilots:
            return []
        
        # Sort îlots by Y coordinate
        sorted_ilots = sorted(ilots, key=lambda x: x['position'][1])
        
        rows = []
        current_row = [sorted_ilots[0]]
        row_tolerance = 2.0  # Tolerance for considering îlots in same row
        
        for i in range(1, len(sorted_ilots)):
            current_y = sorted_ilots[i]['position'][1]
            prev_y = sorted_ilots[i-1]['position'][1]
            
            if abs(current_y - prev_y) <= row_tolerance:
                current_row.append(sorted_ilots[i])
            else:
                if len(current_row) >= 2:  # Only consider rows with multiple îlots
                    rows.append(current_row)
                current_row = [sorted_ilots[i]]
        
        if len(current_row) >= 2:
            rows.append(current_row)
        
        return rows
    
    def _create_corridor_between_rows(self, row1: List[Dict], row2: List[Dict]) -> Optional[Dict]:
        """Create corridor between two rows of îlots"""
        if not row1 or not row2:
            return None
        
        # Calculate corridor bounds
        row1_max_y = max(ilot['bounds'][3] for ilot in row1)  # Top of row1
        row2_min_y = min(ilot['bounds'][1] for ilot in row2)  # Bottom of row2
        
        # Check if rows are close enough for a corridor
        if row2_min_y - row1_max_y < self.corridor_width * 0.5:
            return None
        
        # Calculate corridor position
        corridor_y = (row1_max_y + row2_min_y) / 2
        
        # Find overlapping X range
        row1_min_x = min(ilot['bounds'][0] for ilot in row1)
        row1_max_x = max(ilot['bounds'][2] for ilot in row1)
        row2_min_x = min(ilot['bounds'][0] for ilot in row2)
        row2_max_x = max(ilot['bounds'][2] for ilot in row2)
        
        corridor_min_x = max(row1_min_x, row2_min_x)
        corridor_max_x = min(row1_max_x, row2_max_x)
        
        if corridor_min_x >= corridor_max_x:
            return None
        
        # Create corridor geometry
        corridor_rect = box(
            corridor_min_x, 
            corridor_y - self.corridor_width/2,
            corridor_max_x,
            corridor_y + self.corridor_width/2
        )
        
        return {
            'id': f"corridor_{len(row1)}_{len(row2)}",
            'geometry': corridor_rect,
            'bounds': [corridor_min_x, corridor_y - self.corridor_width/2, 
                      corridor_max_x, corridor_y + self.corridor_width/2],
            'width': self.corridor_width,
            'length': corridor_max_x - corridor_min_x,
            'area': (corridor_max_x - corridor_min_x) * self.corridor_width
        }
    
    def _validate_placement_constraints(self, placed_ilots: List[Dict], 
                                      all_zones: List[Dict], corridors: List[Dict]) -> Dict[str, Any]:
        """Validate that placement respects all constraints"""
        violations = []
        
        # Check distance from entrances/exits
        entrance_zones = [z for z in all_zones if z.get('zone_type') == 'ENTREE_SORTIE']
        
        for ilot in placed_ilots:
            ilot_center = Point(ilot['position'])
            
            for entrance in entrance_zones:
                try:
                    entrance_poly = Polygon(entrance.get('points', []))
                    distance = ilot_center.distance(entrance_poly)
                    
                    if distance < self.min_distance_from_entrance:
                        violations.append({
                            'type': 'entrance_distance',
                            'ilot_id': ilot['id'],
                            'distance': distance,
                            'required': self.min_distance_from_entrance
                        })
                except Exception:
                    continue
        
        # Check for overlaps with restricted zones
        restricted_zones = [z for z in all_zones if z.get('zone_type') == 'NO_ENTREE']
        
        for ilot in placed_ilots:
            ilot_geom = ilot.get('geometry')
            if not ilot_geom:
                continue
                
            for restricted in restricted_zones:
                try:
                    restricted_poly = Polygon(restricted.get('points', []))
                    if ilot_geom.intersects(restricted_poly):
                        violations.append({
                            'type': 'restricted_overlap',
                            'ilot_id': ilot['id'],
                            'restricted_zone': restricted.get('layer', 'unknown')
                        })
                except Exception:
                    continue
        
        return {
            'is_valid': len(violations) == 0,
            'violations': violations,
            'total_violations': len(violations)
        }
    
    def _calculate_achieved_distribution(self, placed_ilots: List[Dict]) -> Dict[str, float]:
        """Calculate the achieved size distribution"""
        if not placed_ilots:
            return {}
        
        size_counts = {}
        total_count = len(placed_ilots)
        
        for ilot in placed_ilots:
            size_range = ilot.get('size_range', 'unknown')
            size_counts[size_range] = size_counts.get(size_range, 0) + 1
        
        # Convert to percentages
        distribution = {}
        for size_range, count in size_counts.items():
            distribution[size_range] = (count / total_count) * 100.0
        
        return distribution
