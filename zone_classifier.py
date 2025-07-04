
"""
Zone Classifier for Îlot Placement
Identifies walls, entrances, restricted areas from architectural plans
"""

import numpy as np
from typing import List, Dict, Any
from shapely.geometry import Polygon, Point


class ZoneClassifier:
    """Classify zones based on architectural plan characteristics"""
    
    def __init__(self):
        self.zone_type_mapping = {
            'walls': ['MUR', 'WALL', 'CLOISON'],
            'entrances': ['ENTREE', 'SORTIE', 'DOOR', 'ENTRANCE', 'EXIT'],
            'restricted': ['ESCALIER', 'ASCENSEUR', 'STAIRS', 'ELEVATOR', 'WC', 'SANITAIRE']
        }
    
    def classify_zones(self, zones: List[Dict]) -> List[Dict]:
        """
        Classify zones into appropriate types for îlot placement
        
        Returns zones with proper zone_type classification:
        - AVAILABLE: Areas where îlots can be placed
        - NO_ENTREE: Restricted areas (stairs, elevators) - light blue
        - ENTREE_SORTIE: Entrances/exits - red
        - MUR: Walls - black (îlots can touch but not overlap)
        """
        classified_zones = []
        
        for zone in zones:
            classified_zone = zone.copy()
            zone_type = self._determine_zone_type(zone)
            classified_zone['zone_type'] = zone_type
            classified_zone['color'] = self._get_zone_color(zone_type)
            classified_zones.append(classified_zone)
        
        return classified_zones
    
    def _determine_zone_type(self, zone: Dict) -> str:
        """Determine zone type based on properties"""
        layer = zone.get('layer', '').upper()
        entity_type = zone.get('entity_type', '').upper()
        area = zone.get('area', 0)
        
        # Check layer names for zone type indicators
        for zone_category, keywords in self.zone_type_mapping.items():
            for keyword in keywords:
                if keyword in layer:
                    if zone_category == 'walls':
                        return 'MUR'
                    elif zone_category == 'entrances':
                        return 'ENTREE_SORTIE'
                    elif zone_category == 'restricted':
                        return 'NO_ENTREE'
        
        # Classify based on area and shape characteristics
        if area < 1.0:  # Very small areas likely restricted or structural
            return 'NO_ENTREE'
        elif area >= 5.0:  # Areas 5m² and above are definitely available for placement
            return 'AVAILABLE'
        else:
            # Medium areas (1-5m²) - analyze shape
            aspect_ratio = self._calculate_aspect_ratio(zone)
            if aspect_ratio > 15:  # Very elongated - likely corridor or wall
                return 'MUR'
            else:
                return 'AVAILABLE'  # Default to available for placement
    
    def _calculate_aspect_ratio(self, zone: Dict) -> float:
        """Calculate aspect ratio of zone"""
        bounds = zone.get('bounds')
        if bounds and len(bounds) >= 4:
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            return max(width, height) / max(min(width, height), 0.1)
        return 1.0
    
    def _get_zone_color(self, zone_type: str) -> str:
        """Get display color for zone type"""
        color_mapping = {
            'AVAILABLE': 'rgba(255, 255, 255, 0.3)',      # White/transparent
            'NO_ENTREE': 'rgba(0, 150, 255, 0.6)',        # Light blue
            'ENTREE_SORTIE': 'rgba(255, 0, 0, 0.7)',      # Red
            'MUR': 'rgba(0, 0, 0, 0.8)'                   # Black
        }
        return color_mapping.get(zone_type, 'rgba(128, 128, 128, 0.5)')
    
    def create_zone_legend(self) -> Dict[str, Dict]:
        """Create legend for zone types"""
        return {
            'NO_ENTREE': {
                'label': 'NO ENTREE',
                'color': 'lightblue',
                'description': 'Restricted areas (stairs, elevators)'
            },
            'ENTREE_SORTIE': {
                'label': 'ENTREE/SORTIE', 
                'color': 'red',
                'description': 'Entrances and exits'
            },
            'MUR': {
                'label': 'MUR',
                'color': 'black',
                'description': 'Walls (îlots can touch)'
            },
            'AVAILABLE': {
                'label': 'Available Space',
                'color': 'white',
                'description': 'Areas for îlot placement'
            }
        }
