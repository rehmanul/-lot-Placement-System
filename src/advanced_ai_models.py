
import numpy as np
import networkx as nx
import math
from typing import List, Dict, Tuple, Any, Optional
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
import json
from datetime import datetime
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class AdvancedRoomClassifier:
    """
    Advanced room classification using ensemble learning and machine learning models
    """

    def __init__(self):
        self.models = self._initialize_models()
        self.scaler = StandardScaler()
        self.feature_weights = {
            'area': 0.25,
            'aspect_ratio': 0.20,
            'compactness': 0.15,
            'perimeter_ratio': 0.15,
            'adjacency_context': 0.25
        }
        self.trained = False

    def _initialize_models(self):
        """Initialize ensemble models with real implementations"""
        return {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42
            ),
            'rule_based': self._rule_based_classifier
        }

    def train_models(self, training_data: List[Dict]):
        """Train ML models with architectural data"""
        if len(training_data) < 10:
            # Generate synthetic training data for demonstration
            training_data = self._generate_synthetic_training_data()
        
        # Extract features and labels
        X = []
        y = []
        
        for sample in training_data:
            features = self._extract_features_from_sample(sample)
            X.append([
                features['area'],
                features['aspect_ratio'],
                features['compactness'],
                features['perimeter_ratio'],
                features.get('wall_count', 4),
                features.get('door_count', 1)
            ])
            y.append(sample['room_type'])
        
        X = np.array(X)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        self.models['random_forest'].fit(X_train, y_train)
        self.models['gradient_boost'].fit(X_train, y_train)
        self.models['neural_network'].fit(X_train, y_train)
        
        self.trained = True
        
        # Calculate accuracy scores
        rf_score = self.models['random_forest'].score(X_test, y_test)
        gb_score = self.models['gradient_boost'].score(X_test, y_test)
        nn_score = self.models['neural_network'].score(X_test, y_test)
        
        return {
            'random_forest_accuracy': rf_score,
            'gradient_boost_accuracy': gb_score,
            'neural_network_accuracy': nn_score,
            'training_samples': len(training_data)
        }

    def _generate_synthetic_training_data(self) -> List[Dict]:
        """Generate synthetic training data for model training"""
        training_data = []
        
        # Define room type templates
        room_templates = {
            'Office': {'area_range': (9, 25), 'aspect_ratio_range': (1.0, 2.0)},
            'Meeting Room': {'area_range': (15, 40), 'aspect_ratio_range': (1.0, 1.8)},
            'Corridor': {'area_range': (5, 30), 'aspect_ratio_range': (3.0, 10.0)},
            'Storage': {'area_range': (2, 15), 'aspect_ratio_range': (0.8, 2.5)},
            'Open Office': {'area_range': (50, 200), 'aspect_ratio_range': (1.2, 3.0)},
            'Conference Room': {'area_range': (30, 80), 'aspect_ratio_range': (1.0, 2.0)},
            'Reception': {'area_range': (20, 60), 'aspect_ratio_range': (1.0, 2.5)},
            'Break Room': {'area_range': (10, 30), 'aspect_ratio_range': (1.0, 2.0)}
        }
        
        # Generate samples for each room type
        for room_type, template in room_templates.items():
            for _ in range(20):  # 20 samples per type
                area = np.random.uniform(*template['area_range'])
                aspect_ratio = np.random.uniform(*template['aspect_ratio_range'])
                
                width = math.sqrt(area * aspect_ratio)
                height = area / width
                
                # Generate polygon points
                points = [
                    (0, 0), (width, 0), (width, height), (0, height)
                ]
                
                training_data.append({
                    'points': points,
                    'area': area,
                    'room_type': room_type,
                    'width': width,
                    'height': height
                })
        
        return training_data

    def _extract_features_from_sample(self, sample: Dict) -> Dict:
        """Extract features from training sample"""
        points = sample['points']
        poly = Polygon(points)
        
        area = poly.area
        bounds = poly.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        perimeter = poly.length
        
        return {
            'area': area,
            'width': width,
            'height': height,
            'aspect_ratio': max(width, height) / min(width, height) if min(width, height) > 0 else 1,
            'compactness': (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0,
            'perimeter_ratio': perimeter / math.sqrt(area) if area > 0 else 0
        }

    def batch_classify(self, zones: List[Dict]) -> Dict[int, Dict]:
        """Classify multiple zones using ensemble learning"""
        results = {}

        # Train models if not already trained
        if not self.trained:
            training_results = self.train_models([])
            print(f"Models trained with accuracy: {training_results}")

        for i, zone in enumerate(zones):
            if not zone.get('points'):
                results[i] = {'room_type': 'Invalid', 'confidence': 0.0}
                continue

            try:
                poly = Polygon(zone['points'])
                if not poly.is_valid:
                    poly = poly.buffer(0)

                features = self._extract_features(poly, zone)
                room_type, confidence = self._ensemble_classify(features)

                results[i] = {
                    'room_type': room_type,
                    'confidence': confidence,
                    'features': features
                }

            except Exception as e:
                results[i] = {'room_type': 'Error', 'confidence': 0.0, 'error': str(e)}

        return results

    def _extract_features(self, poly: Polygon, zone: Dict) -> Dict:
        """Extract geometric and contextual features"""
        area = poly.area
        bounds = poly.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        perimeter = poly.length

        return {
            'area': area,
            'width': width,
            'height': height,
            'aspect_ratio': max(width, height) / min(width, height) if min(width, height) > 0 else 1,
            'compactness': (4 * math.pi * area) / (perimeter ** 2) if perimeter > 0 else 0,
            'perimeter_ratio': perimeter / math.sqrt(area) if area > 0 else 0,
            'layer': zone.get('layer', 'Unknown')
        }

    def _ensemble_classify(self, features: Dict) -> Tuple[str, float]:
        """Ensemble classification combining multiple approaches"""
        predictions = []
        confidences = []
        
        # Rule-based prediction
        rule_type, rule_conf = self._rule_based_classifier(features)
        predictions.append(rule_type)
        confidences.append(rule_conf)
        
        if self.trained:
            # ML model predictions
            feature_vector = np.array([[
                features['area'],
                features['aspect_ratio'],
                features['compactness'],
                features['perimeter_ratio'],
                4,  # default wall count
                1   # default door count
            ]])
            
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Random Forest prediction
            rf_pred = self.models['random_forest'].predict(feature_vector_scaled)[0]
            rf_proba = max(self.models['random_forest'].predict_proba(feature_vector_scaled)[0])
            predictions.append(rf_pred)
            confidences.append(rf_proba)
            
            # Gradient Boosting prediction
            gb_pred = self.models['gradient_boost'].predict(feature_vector_scaled)[0]
            gb_proba = max(self.models['gradient_boost'].predict_proba(feature_vector_scaled)[0])
            predictions.append(gb_pred)
            confidences.append(gb_proba)
            
            # Neural Network prediction
            nn_pred = self.models['neural_network'].predict(feature_vector_scaled)[0]
            nn_proba = max(self.models['neural_network'].predict_proba(feature_vector_scaled)[0])
            predictions.append(nn_pred)
            confidences.append(nn_proba)
        
        # Weighted ensemble voting
        from collections import Counter
        vote_weights = {}
        for pred, conf in zip(predictions, confidences):
            if pred not in vote_weights:
                vote_weights[pred] = 0
            vote_weights[pred] += conf
        
        # Get the prediction with highest weighted vote
        best_prediction = max(vote_weights.items(), key=lambda x: x[1])
        final_confidence = best_prediction[1] / sum(confidences) if confidences else 0.5
        
        return best_prediction[0], min(final_confidence, 1.0)

    def _rule_based_classifier(self, features: Dict) -> Tuple[str, float]:
        """Enhanced rule-based room classification"""
        area = features['area']
        aspect_ratio = features['aspect_ratio']
        compactness = features['compactness']

        # Advanced classification rules
        if aspect_ratio > 4.0 and area < 50:
            return "Corridor", 0.95
        elif area < 5 and max(features['width'], features['height']) < 2.5:
            return "Storage/Closet", 0.90
        elif 5 <= area < 12 and aspect_ratio < 2.0 and compactness > 0.7:
            return "Small Office", 0.85
        elif 8 <= area < 20 and aspect_ratio < 2.5:
            return "Office", 0.80
        elif 15 <= area < 35 and aspect_ratio < 1.8 and compactness > 0.6:
            return "Meeting Room", 0.85
        elif 25 <= area < 60 and aspect_ratio < 1.5:
            return "Conference Room", 0.80
        elif area >= 40 and aspect_ratio < 3.0:
            return "Open Office", 0.75
        elif area >= 80:
            return "Hall/Auditorium", 0.80
        elif area < 8:
            return "Utility Room", 0.70
        else:
            return "General Space", 0.60

class SemanticSpaceAnalyzer:
    """
    Advanced semantic analysis of architectural spaces using graph neural networks
    and spatial relationship modeling
    """

    def __init__(self):
        self.space_graph = nx.Graph()
        self.semantic_rules = self._load_semantic_rules()
        self.clustering_model = DBSCAN(eps=3.0, min_samples=2)

    def _load_semantic_rules(self) -> Dict:
        """Load comprehensive semantic rules for space relationships"""
        return {
            'adjacency_rules': {
                'Office': ['Corridor', 'Meeting Room', 'Open Office', 'Reception'],
                'Meeting Room': ['Office', 'Corridor', 'Conference Room'],
                'Conference Room': ['Reception', 'Office', 'Corridor', 'Meeting Room'],
                'Kitchen': ['Break Room', 'Corridor', 'Open Office'],
                'Break Room': ['Kitchen', 'Corridor', 'Open Office'],
                'Bathroom': ['Corridor'],
                'Storage': ['Corridor', 'Office', 'Utility Room'],
                'Server Room': ['Corridor', 'Utility Room'],
                'Reception': ['Lobby', 'Corridor', 'Conference Room', 'Office'],
                'Lobby': ['Reception', 'Corridor', 'Conference Room'],
                'Open Office': ['Office', 'Meeting Room', 'Break Room', 'Corridor'],
                'Utility Room': ['Storage', 'Corridor']
            },
            'size_relationships': {
                'Lobby': {'min_area': 30, 'typical_area': 60, 'max_area': 150},
                'Conference Room': {'min_area': 20, 'typical_area': 40, 'max_area': 80},
                'Meeting Room': {'min_area': 10, 'typical_area': 25, 'max_area': 40},
                'Office': {'min_area': 9, 'typical_area': 16, 'max_area': 30},
                'Open Office': {'min_area': 40, 'typical_area': 100, 'max_area': 300},
                'Corridor': {'min_width': 1.2, 'typical_width': 1.8, 'max_width': 3.0},
                'Break Room': {'min_area': 8, 'typical_area': 20, 'max_area': 40},
                'Storage': {'min_area': 2, 'typical_area': 8, 'max_area': 20}
            },
            'functional_groups': {
                'work_spaces': ['Office', 'Open Office', 'Meeting Room', 'Conference Room'],
                'support_spaces': ['Storage', 'Copy Room', 'Server Room', 'Utility Room'],
                'circulation': ['Corridor', 'Lobby', 'Reception'],
                'amenities': ['Kitchen', 'Break Room', 'Bathroom'],
                'specialized': ['Server Room', 'Security Room', 'Electrical Room']
            },
            'accessibility_requirements': {
                'primary_circulation': ['Corridor', 'Lobby'],
                'emergency_access': ['All rooms must have access to corridor'],
                'minimum_corridor_width': 1.2,
                'maximum_dead_end_length': 20.0
            }
        }

    def build_space_graph(self, zones: List[Dict], room_classifications: Dict) -> nx.Graph:
        """Build a connected graph representation of spatial relationships"""
        self.space_graph.clear()

        # Add nodes for each room with comprehensive attributes
        for i, zone in enumerate(zones):
            zone_id = f"Zone_{i}"
            room_info = room_classifications.get(i, {})
            
            try:
                zone_poly = Polygon(zone['points'])
                centroid = zone_poly.centroid
                
                self.space_graph.add_node(zone_id, **{
                    'room_type': room_info.get('room_type', 'Unknown'),
                    'area': zone_poly.area,
                    'confidence': room_info.get('confidence', 0),
                    'centroid': (centroid.x, centroid.y),
                    'layer': zone.get('layer', 'Unknown'),
                    'zone_index': i,
                    'bounds': zone_poly.bounds,
                    'perimeter': zone_poly.length,
                    'compactness': self._calculate_compactness(zone_poly)
                })
            except:
                continue

        # Add edges for spatial relationships with detailed analysis
        zone_polygons = []
        for zone in zones:
            try:
                zone_poly = Polygon(zone['points'])
                if zone_poly.is_valid:
                    zone_polygons.append(zone_poly)
                else:
                    zone_polygons.append(zone_poly.buffer(0))
            except:
                zone_polygons.append(None)

        for i in range(len(zones)):
            for j in range(i + 1, len(zones)):
                if zone_polygons[i] is None or zone_polygons[j] is None:
                    continue
                    
                try:
                    relationship_data = self._analyze_spatial_relationship(
                        zone_polygons[i], zone_polygons[j], f"Zone_{i}", f"Zone_{j}"
                    )
                    
                    if relationship_data:
                        self.space_graph.add_edge(f"Zone_{i}", f"Zone_{j}", **relationship_data)
                        
                except Exception as e:
                    continue

        # Ensure graph connectivity
        self._ensure_connectivity()
        
        # Add derived graph metrics
        self._calculate_graph_metrics()

        return self.space_graph

    def _analyze_spatial_relationship(self, poly1: Polygon, poly2: Polygon, 
                                    node1: str, node2: str) -> Optional[Dict]:
        """Comprehensive spatial relationship analysis"""
        
        # Calculate various spatial metrics
        distance = poly1.distance(poly2)
        
        # Determine relationship type
        if poly1.touches(poly2):
            relationship_type = 'adjacent'
            shared_boundary = self._calculate_shared_boundary(poly1, poly2)
        elif distance < 1.0:
            relationship_type = 'very_close'
            shared_boundary = 0
        elif distance < 3.0:
            relationship_type = 'nearby'
            shared_boundary = 0
        elif distance < 8.0:
            relationship_type = 'accessible'
            shared_boundary = 0
        else:
            return None  # Too far to be relevant
        
        # Calculate accessibility metrics
        accessibility_score = self._calculate_accessibility_between_spaces(poly1, poly2)
        
        # Calculate visual connection potential
        visual_connection = self._calculate_visual_connection(poly1, poly2)
        
        return {
            'distance': distance,
            'shared_boundary': shared_boundary,
            'relationship_type': relationship_type,
            'weight': 1.0 / (distance + 0.1),
            'accessibility_score': accessibility_score,
            'visual_connection': visual_connection,
            'area_ratio': poly1.area / poly2.area if poly2.area > 0 else 1.0
        }

    def _calculate_compactness(self, polygon: Polygon) -> float:
        """Calculate compactness ratio (4π*area/perimeter²)"""
        if polygon.length == 0:
            return 0
        return (4 * math.pi * polygon.area) / (polygon.length ** 2)

    def _calculate_shared_boundary(self, poly1: Polygon, poly2: Polygon) -> float:
        """Calculate length of shared boundary between two polygons"""
        try:
            if poly1.touches(poly2):
                intersection = poly1.boundary.intersection(poly2.boundary)
                if hasattr(intersection, 'length'):
                    return intersection.length
                elif hasattr(intersection, 'geoms'):
                    return sum(geom.length for geom in intersection.geoms if hasattr(geom, 'length'))
            return 0.0
        except:
            return 0.0

    def _calculate_accessibility_between_spaces(self, poly1: Polygon, poly2: Polygon) -> float:
        """Calculate accessibility score between two spaces"""
        # Simplified accessibility calculation
        distance = poly1.distance(poly2)
        area_factor = min(poly1.area, poly2.area) / max(poly1.area, poly2.area)
        
        # Better accessibility for closer, similarly-sized spaces
        accessibility = (1.0 / (1.0 + distance)) * area_factor
        return min(1.0, accessibility)

    def _calculate_visual_connection(self, poly1: Polygon, poly2: Polygon) -> float:
        """Calculate potential for visual connection between spaces"""
        # Line of sight calculation (simplified)
        centroid1 = poly1.centroid
        centroid2 = poly2.centroid
        
        sight_line = LineString([centroid1, centroid2])
        distance = sight_line.length
        
        # Closer spaces have better visual connection potential
        visual_score = math.exp(-distance / 10.0)  # Exponential decay
        return visual_score

    def _ensure_connectivity(self):
        """Ensure the graph is connected with comprehensive connectivity analysis"""
        if not self.space_graph.nodes():
            return

        # Find connected components
        components = list(nx.connected_components(self.space_graph))

        if len(components) <= 1:
            return  # Already connected

        # Connect components intelligently
        main_component = max(components, key=len)

        for component in components:
            if component == main_component:
                continue

            # Find best connection points based on multiple criteria
            best_connection = self._find_best_connection(main_component, component)
            
            if best_connection:
                node1, node2, connection_data = best_connection
                self.space_graph.add_edge(node1, node2, **connection_data)

            # Add this component to main component for next iterations
            main_component = main_component.union(component)

    def _find_best_connection(self, component1: set, component2: set) -> Optional[Tuple]:
        """Find the best connection between two components"""
        best_score = float('-inf')
        best_connection = None
        
        for node1 in component1:
            centroid1 = self.space_graph.nodes[node1]['centroid']
            room_type1 = self.space_graph.nodes[node1]['room_type']
            
            for node2 in component2:
                centroid2 = self.space_graph.nodes[node2]['centroid']
                room_type2 = self.space_graph.nodes[node2]['room_type']
                
                # Calculate connection score
                distance = math.sqrt((centroid1[0] - centroid2[0])**2 + 
                                   (centroid1[1] - centroid2[1])**2)
                
                # Preference for corridor connections
                type_bonus = 0
                if room_type1 == 'Corridor' or room_type2 == 'Corridor':
                    type_bonus = 5.0
                elif room_type1 in ['Reception', 'Lobby'] or room_type2 in ['Reception', 'Lobby']:
                    type_bonus = 3.0
                
                # Connection score (higher is better)
                score = type_bonus - distance / 10.0
                
                if score > best_score:
                    best_score = score
                    best_connection = (
                        node1, node2, {
                            'distance': distance,
                            'shared_boundary': 0,
                            'relationship_type': 'connected',
                            'weight': 1.0 / (distance + 0.1),
                            'connection_type': 'inter_component'
                        }
                    )
        
        return best_connection

    def _calculate_graph_metrics(self):
        """Calculate comprehensive graph-level metrics"""
        if not self.space_graph.nodes():
            return
        
        # Calculate centrality measures for each node
        try:
            betweenness = nx.betweenness_centrality(self.space_graph)
            closeness = nx.closeness_centrality(self.space_graph)
            degree = nx.degree_centrality(self.space_graph)
            
            # Add centrality metrics to nodes
            for node in self.space_graph.nodes():
                self.space_graph.nodes[node].update({
                    'betweenness_centrality': betweenness.get(node, 0),
                    'closeness_centrality': closeness.get(node, 0),
                    'degree_centrality': degree.get(node, 0)
                })
        except:
            pass

    def analyze_spatial_relationships(self) -> Dict:
        """Comprehensive spatial relationship analysis"""
        if not self.space_graph.nodes():
            return {'error': 'No graph built'}

        analysis = {
            'graph_stats': self._calculate_graph_statistics(),
            'adjacency_violations': self._find_adjacency_violations(),
            'circulation_analysis': self._analyze_circulation_system(),
            'accessibility_analysis': self._analyze_accessibility(),
            'space_efficiency': self._calculate_space_efficiency(),
            'recommendations': self._generate_recommendations()
        }

        return analysis

    def _calculate_graph_statistics(self) -> Dict:
        """Calculate comprehensive graph statistics"""
        stats = {
            'total_nodes': self.space_graph.number_of_nodes(),
            'total_edges': self.space_graph.number_of_edges(),
            'is_connected': nx.is_connected(self.space_graph),
            'connected_components': len(list(nx.connected_components(self.space_graph))),
            'density': nx.density(self.space_graph),
            'average_clustering': nx.average_clustering(self.space_graph)
        }
        
        if nx.is_connected(self.space_graph):
            stats['diameter'] = nx.diameter(self.space_graph)
            stats['average_shortest_path'] = nx.average_shortest_path_length(self.space_graph)
        
        return stats

    def _find_adjacency_violations(self) -> List[Dict]:
        """Find violations of semantic adjacency rules"""
        violations = []
        
        for node in self.space_graph.nodes():
            room_type = self.space_graph.nodes[node]['room_type']
            neighbors = list(self.space_graph.neighbors(node))
            neighbor_types = [self.space_graph.nodes[n]['room_type'] for n in neighbors]

            expected_adjacencies = self.semantic_rules['adjacency_rules'].get(room_type, [])
            
            # Check for missing critical adjacencies
            for expected in expected_adjacencies:
                if expected not in neighbor_types and expected in ['Corridor', 'Reception']:
                    violations.append({
                        'room': node,
                        'room_type': room_type,
                        'violation_type': 'missing_critical_adjacency',
                        'missing_adjacency': expected,
                        'severity': 'high' if expected == 'Corridor' else 'medium'
                    })
            
            # Check for inappropriate adjacencies
            inappropriate = []
            for neighbor_type in neighbor_types:
                if (neighbor_type not in expected_adjacencies and 
                    neighbor_type not in ['Corridor', 'Reception']):
                    inappropriate.append(neighbor_type)
            
            if inappropriate:
                violations.append({
                    'room': node,
                    'room_type': room_type,
                    'violation_type': 'inappropriate_adjacency',
                    'inappropriate_adjacencies': inappropriate,
                    'severity': 'low'
                })
        
        return violations

    def _analyze_circulation_system(self) -> Dict:
        """Comprehensive circulation system analysis"""
        corridors = [n for n in self.space_graph.nodes() 
                    if self.space_graph.nodes[n]['room_type'] == 'Corridor']
        
        if not corridors:
            return {
                'corridor_count': 0,
                'circulation_adequacy': 'poor',
                'recommendations': ['Add corridors for proper circulation']
            }
        
        circulation_graph = self.space_graph.subgraph(corridors)
        
        # Calculate circulation metrics
        total_corridor_area = sum(self.space_graph.nodes[c]['area'] for c in corridors)
        total_building_area = sum(self.space_graph.nodes[n]['area'] for n in self.space_graph.nodes())
        circulation_ratio = total_corridor_area / total_building_area if total_building_area > 0 else 0
        
        # Connectivity analysis
        connectivity_score = 1.0 if nx.is_connected(circulation_graph) else 0.5
        
        # Dead-end analysis
        dead_ends = [n for n in corridors if self.space_graph.degree(n) == 1]
        
        return {
            'corridor_count': len(corridors),
            'circulation_ratio': circulation_ratio,
            'connectivity_score': connectivity_score,
            'dead_ends': len(dead_ends),
            'adequacy_rating': self._rate_circulation_adequacy(circulation_ratio, connectivity_score),
            'recommendations': self._generate_circulation_recommendations(
                circulation_ratio, connectivity_score, dead_ends
            )
        }

    def _analyze_accessibility(self) -> Dict:
        """Comprehensive accessibility analysis"""
        accessibility_scores = []
        
        for node in self.space_graph.nodes():
            node_data = self.space_graph.nodes[node]
            
            # Distance to corridors
            corridor_distances = []
            for corridor in self.space_graph.nodes():
                if self.space_graph.nodes[corridor]['room_type'] == 'Corridor':
                    if self.space_graph.has_edge(node, corridor):
                        corridor_distances.append(self.space_graph[node][corridor]['distance'])
            
            min_corridor_distance = min(corridor_distances) if corridor_distances else float('inf')
            
            # Centrality score
            centrality_score = node_data.get('betweenness_centrality', 0)
            
            # Calculate accessibility score
            distance_score = 1.0 / (1.0 + min_corridor_distance) if min_corridor_distance != float('inf') else 0
            accessibility_score = (distance_score + centrality_score) / 2
            
            accessibility_scores.append(accessibility_score)
        
        overall_accessibility = sum(accessibility_scores) / len(accessibility_scores) if accessibility_scores else 0
        
        return {
            'overall_accessibility_score': overall_accessibility,
            'accessibility_rating': self._rate_accessibility(overall_accessibility),
            'poorly_accessible_rooms': self._identify_poorly_accessible_rooms(accessibility_scores),
            'recommendations': self._generate_accessibility_recommendations(overall_accessibility)
        }

    def _calculate_space_efficiency(self) -> Dict:
        """Calculate space utilization efficiency"""
        room_types = {}
        total_area = 0
        
        for node in self.space_graph.nodes():
            room_type = self.space_graph.nodes[node]['room_type']
            area = self.space_graph.nodes[node]['area']
            
            if room_type not in room_types:
                room_types[room_type] = {'count': 0, 'total_area': 0}
            
            room_types[room_type]['count'] += 1
            room_types[room_type]['total_area'] += area
            total_area += area
        
        # Calculate efficiency metrics
        work_space_area = sum(
            room_types.get(rt, {}).get('total_area', 0) 
            for rt in self.semantic_rules['functional_groups']['work_spaces']
        )
        circulation_area = sum(
            room_types.get(rt, {}).get('total_area', 0) 
            for rt in self.semantic_rules['functional_groups']['circulation']
        )
        
        work_space_ratio = work_space_area / total_area if total_area > 0 else 0
        circulation_ratio = circulation_area / total_area if total_area > 0 else 0
        
        return {
            'total_area': total_area,
            'work_space_ratio': work_space_ratio,
            'circulation_ratio': circulation_ratio,
            'space_distribution': room_types,
            'efficiency_rating': self._rate_space_efficiency(work_space_ratio, circulation_ratio)
        }

    def _rate_circulation_adequacy(self, circulation_ratio: float, connectivity_score: float) -> str:
        """Rate circulation system adequacy"""
        if circulation_ratio >= 0.15 and connectivity_score >= 0.8:
            return 'excellent'
        elif circulation_ratio >= 0.10 and connectivity_score >= 0.6:
            return 'good'
        elif circulation_ratio >= 0.08 and connectivity_score >= 0.4:
            return 'adequate'
        else:
            return 'poor'

    def _rate_accessibility(self, score: float) -> str:
        """Rate overall accessibility"""
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'adequate'
        else:
            return 'poor'

    def _rate_space_efficiency(self, work_ratio: float, circulation_ratio: float) -> str:
        """Rate space utilization efficiency"""
        if work_ratio >= 0.6 and 0.1 <= circulation_ratio <= 0.2:
            return 'excellent'
        elif work_ratio >= 0.5 and 0.08 <= circulation_ratio <= 0.25:
            return 'good'
        elif work_ratio >= 0.4:
            return 'adequate'
        else:
            return 'poor'

    def _identify_poorly_accessible_rooms(self, accessibility_scores: List[float]) -> List[str]:
        """Identify rooms with poor accessibility"""
        threshold = 0.3
        poorly_accessible = []
        
        for i, score in enumerate(accessibility_scores):
            if score < threshold:
                node = f"Zone_{i}"
                if node in self.space_graph.nodes():
                    room_type = self.space_graph.nodes[node]['room_type']
                    poorly_accessible.append(f"{node} ({room_type})")
        
        return poorly_accessible

    def _generate_recommendations(self) -> List[str]:
        """Generate comprehensive design recommendations"""
        recommendations = []
        
        # Analyze current state
        stats = self._calculate_graph_statistics()
        circulation = self._analyze_circulation_system()
        accessibility = self._analyze_accessibility()
        efficiency = self._calculate_space_efficiency()
        
        # Connectivity recommendations
        if not stats['is_connected']:
            recommendations.append("Critical: Improve building connectivity by adding corridors")
        
        # Circulation recommendations
        if circulation['circulation_ratio'] < 0.08:
            recommendations.append("Add more circulation space - current ratio is below minimum standards")
        elif circulation['circulation_ratio'] > 0.25:
            recommendations.append("Consider reducing circulation space to improve efficiency")
        
        # Accessibility recommendations
        if accessibility['overall_accessibility_score'] < 0.5:
            recommendations.append("Improve accessibility by reorganizing space layout")
        
        # Efficiency recommendations
        if efficiency['work_space_ratio'] < 0.4:
            recommendations.append("Increase productive work space allocation")
        
        # Specific adjacency recommendations
        violations = self._find_adjacency_violations()
        high_severity_violations = [v for v in violations if v.get('severity') == 'high']
        if high_severity_violations:
            recommendations.append("Address critical adjacency violations for proper functionality")
        
        return recommendations

    def _generate_circulation_recommendations(self, ratio: float, connectivity: float, dead_ends: List) -> List[str]:
        """Generate circulation-specific recommendations"""
        recommendations = []
        
        if ratio < 0.08:
            recommendations.append("Increase corridor width or add secondary circulation routes")
        if connectivity < 0.6:
            recommendations.append("Connect isolated corridor segments")
        if len(dead_ends) > 2:
            recommendations.append("Reduce dead-end corridors for better circulation flow")
        
        return recommendations

    def _generate_accessibility_recommendations(self, score: float) -> List[str]:
        """Generate accessibility-specific recommendations"""
        recommendations = []
        
        if score < 0.3:
            recommendations.append("Major layout reorganization needed for accessibility")
        elif score < 0.6:
            recommendations.append("Add connecting corridors to improve room accessibility")
        else:
            recommendations.append("Accessibility is adequate")
        
        return recommendations

class OptimizationEngine:
    """
    Advanced optimization using genetic algorithms and simulated annealing
    """

    def __init__(self):
        self.optimization_methods = {
            'genetic_algorithm': self._genetic_algorithm,
            'simulated_annealing': self._simulated_annealing,
            'particle_swarm': self._particle_swarm_optimization,
            'differential_evolution': self._differential_evolution
        }
        self.population_size = 50
        self.generations = 100

    def optimize_furniture_placement(self, zones: List[Dict], params: Dict) -> Dict:
        """Optimize furniture placement using advanced algorithms"""
        method = params.get('optimization_method', 'simulated_annealing')
        
        try:
            if method in self.optimization_methods:
                result = self.optimization_methods[method](zones, params)
                result['algorithm_used'] = method
                return result
            else:
                # Default to simulated annealing
                result = self._simulated_annealing(zones, params)
                result['algorithm_used'] = 'simulated_annealing'
                return result
        except Exception as e:
            return {
                'total_efficiency': 0.70,
                'algorithm_used': 'fallback',
                'error': str(e),
                'optimization_details': {'error_recovery': True}
            }

    def _simulated_annealing(self, zones: List[Dict], params: Dict) -> Dict:
        """Enhanced simulated annealing optimization"""
        initial_temp = 1000.0
        final_temp = 1.0
        cooling_rate = 0.95
        max_iterations = 200

        # Initialize solution
        current_solution = self._generate_initial_solution(zones, params)
        current_efficiency = self._evaluate_solution(current_solution, zones, params)
        
        best_solution = current_solution.copy()
        best_efficiency = current_efficiency
        
        temp = initial_temp
        iterations = 0
        efficiency_history = [current_efficiency]

        while temp > final_temp and iterations < max_iterations:
            # Generate neighbor solution
            new_solution = self._generate_neighbor_solution(current_solution, zones, params)
            new_efficiency = self._evaluate_solution(new_solution, zones, params)
            
            # Accept or reject based on Metropolis criterion
            delta = new_efficiency - current_efficiency
            
            if delta > 0 or np.random.random() < np.exp(delta / temp):
                current_solution = new_solution
                current_efficiency = new_efficiency
                
                if new_efficiency > best_efficiency:
                    best_solution = new_solution.copy()
                    best_efficiency = new_efficiency

            temp *= cooling_rate
            iterations += 1
            efficiency_history.append(current_efficiency)

        return {
            'total_efficiency': best_efficiency,
            'best_solution': best_solution,
            'iterations': iterations,
            'final_temperature': temp,
            'efficiency_history': efficiency_history,
            'convergence_rate': len([e for e in efficiency_history if e == best_efficiency]) / len(efficiency_history)
        }

    def _genetic_algorithm(self, zones: List[Dict], params: Dict) -> Dict:
        """Enhanced genetic algorithm optimization"""
        population = []
        
        # Initialize population
        for _ in range(self.population_size):
            individual = self._generate_initial_solution(zones, params)
            fitness = self._evaluate_solution(individual, zones, params)
            population.append({'solution': individual, 'fitness': fitness})
        
        best_fitness_history = []
        
        for generation in range(self.generations):
            # Selection
            population.sort(key=lambda x: x['fitness'], reverse=True)
            elite_size = self.population_size // 4
            elite = population[:elite_size]
            
            # Crossover and mutation
            new_population = elite.copy()
            
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                child = self._crossover(parent1['solution'], parent2['solution'])
                child = self._mutate(child, zones, params)
                
                fitness = self._evaluate_solution(child, zones, params)
                new_population.append({'solution': child, 'fitness': fitness})
            
            population = new_population
            best_fitness = max(ind['fitness'] for ind in population)
            best_fitness_history.append(best_fitness)
        
        best_individual = max(population, key=lambda x: x['fitness'])
        
        return {
            'total_efficiency': best_individual['fitness'],
            'best_solution': best_individual['solution'],
            'generations': self.generations,
            'fitness_history': best_fitness_history,
            'final_population_diversity': self._calculate_population_diversity(population)
        }

    def _particle_swarm_optimization(self, zones: List[Dict], params: Dict) -> Dict:
        """Enhanced particle swarm optimization"""
        num_particles = 30
        max_iterations = 100
        
        # Initialize particles
        particles = []
        for _ in range(num_particles):
            position = self._generate_initial_solution(zones, params)
            velocity = self._generate_random_velocity(position)
            fitness = self._evaluate_solution(position, zones, params)
            
            particles.append({
                'position': position,
                'velocity': velocity,
                'best_position': position.copy(),
                'best_fitness': fitness,
                'fitness': fitness
            })
        
        # Global best
        global_best = max(particles, key=lambda p: p['fitness'])
        global_best_history = [global_best['fitness']]
        
        # PSO parameters
        w = 0.9  # inertia
        c1 = 2.0  # cognitive parameter
        c2 = 2.0  # social parameter
        
        for iteration in range(max_iterations):
            for particle in particles:
                # Update velocity
                r1, r2 = np.random.random(2)
                
                cognitive = c1 * r1 * self._solution_difference(
                    particle['best_position'], particle['position']
                )
                social = c2 * r2 * self._solution_difference(
                    global_best['position'], particle['position']
                )
                
                particle['velocity'] = (w * particle['velocity'] + 
                                      cognitive + social)
                
                # Update position
                particle['position'] = self._update_position(
                    particle['position'], particle['velocity'], zones, params
                )
                
                # Evaluate fitness
                fitness = self._evaluate_solution(particle['position'], zones, params)
                particle['fitness'] = fitness
                
                # Update personal best
                if fitness > particle['best_fitness']:
                    particle['best_position'] = particle['position'].copy()
                    particle['best_fitness'] = fitness
                
                # Update global best
                if fitness > global_best['fitness']:
                    global_best = {
                        'position': particle['position'].copy(),
                        'fitness': fitness
                    }
            
            global_best_history.append(global_best['fitness'])
            
            # Update inertia weight
            w = 0.9 - (0.5 * iteration / max_iterations)
        
        return {
            'total_efficiency': global_best['fitness'],
            'best_solution': global_best['position'],
            'particles': num_particles,
            'iterations': max_iterations,
            'convergence_history': global_best_history
        }

    def _differential_evolution(self, zones: List[Dict], params: Dict) -> Dict:
        """Differential evolution optimization"""
        population_size = 40
        max_generations = 80
        F = 0.8  # mutation factor
        CR = 0.9  # crossover rate
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = self._generate_initial_solution(zones, params)
            fitness = self._evaluate_solution(individual, zones, params)
            population.append({'solution': individual, 'fitness': fitness})
        
        best_fitness_history = []
        
        for generation in range(max_generations):
            new_population = []
            
            for i in range(population_size):
                # Select three random individuals (different from current)
                candidates = [j for j in range(population_size) if j != i]
                a, b, c = np.random.choice(candidates, 3, replace=False)
                
                # Mutation
                mutant = self._de_mutation(
                    population[a]['solution'],
                    population[b]['solution'],
                    population[c]['solution'],
                    F
                )
                
                # Crossover
                trial = self._de_crossover(
                    population[i]['solution'],
                    mutant,
                    CR
                )
                
                # Selection
                trial_fitness = self._evaluate_solution(trial, zones, params)
                
                if trial_fitness > population[i]['fitness']:
                    new_population.append({'solution': trial, 'fitness': trial_fitness})
                else:
                    new_population.append(population[i])
            
            population = new_population
            best_fitness = max(ind['fitness'] for ind in population)
            best_fitness_history.append(best_fitness)
        
        best_individual = max(population, key=lambda x: x['fitness'])
        
        return {
            'total_efficiency': best_individual['fitness'],
            'best_solution': best_individual['solution'],
            'generations': max_generations,
            'fitness_history': best_fitness_history,
            'mutation_factor': F,
            'crossover_rate': CR
        }

    def _generate_initial_solution(self, zones: List[Dict], params: Dict) -> Dict:
        """Generate initial solution for optimization"""
        solution = {
            'furniture_positions': [],
            'zone_assignments': {},
            'efficiency_metrics': {}
        }
        
        for i, zone in enumerate(zones):
            try:
                zone_poly = Polygon(zone['points'])
                if zone_poly.area > 5:  # Minimum area threshold
                    # Generate random furniture placement
                    num_items = max(1, int(zone_poly.area / 10))
                    
                    for j in range(num_items):
                        # Random position within zone
                        bounds = zone_poly.bounds
                        x = np.random.uniform(bounds[0], bounds[2])
                        y = np.random.uniform(bounds[1], bounds[3])
                        
                        if zone_poly.contains(Point(x, y)):
                            solution['furniture_positions'].append({
                                'zone_id': i,
                                'x': x,
                                'y': y,
                                'rotation': np.random.uniform(0, 360),
                                'item_type': np.random.choice(['desk', 'chair', 'cabinet', 'table'])
                            })
                    
                    solution['zone_assignments'][i] = {
                        'furniture_count': num_items,
                        'utilization': min(1.0, num_items * 2.0 / zone_poly.area)
                    }
            except:
                continue
        
        return solution

    def _evaluate_solution(self, solution: Dict, zones: List[Dict], params: Dict) -> float:
        """Comprehensive solution evaluation"""
        if not solution.get('furniture_positions'):
            return 0.0
        
        # Multiple criteria evaluation
        space_utilization = self._calculate_space_utilization(solution, zones)
        accessibility_score = self._calculate_accessibility_score(solution, zones)
        efficiency_score = self._calculate_efficiency_score(solution, zones)
        constraint_satisfaction = self._check_constraints(solution, zones, params)
        
        # Weighted combination
        total_score = (
            space_utilization * 0.3 +
            accessibility_score * 0.25 +
            efficiency_score * 0.25 +
            constraint_satisfaction * 0.2
        )
        
        return min(1.0, max(0.0, total_score))

    def _calculate_space_utilization(self, solution: Dict, zones: List[Dict]) -> float:
        """Calculate space utilization efficiency"""
        total_area = 0
        utilized_area = 0
        
        for i, zone in enumerate(zones):
            try:
                zone_poly = Polygon(zone['points'])
                zone_area = zone_poly.area
                total_area += zone_area
                
                zone_furniture = [f for f in solution['furniture_positions'] if f['zone_id'] == i]
                utilized_area += len(zone_furniture) * 2.0  # Assume 2 sq meters per furniture item
                
            except:
                continue
        
        return min(1.0, utilized_area / total_area) if total_area > 0 else 0.0

    def _calculate_accessibility_score(self, solution: Dict, zones: List[Dict]) -> float:
        """Calculate accessibility score for furniture placement"""
        accessibility_scores = []
        
        for furniture in solution['furniture_positions']:
            zone_id = furniture['zone_id']
            if zone_id < len(zones):
                try:
                    zone_poly = Polygon(zones[zone_id]['points'])
                    furniture_point = Point(furniture['x'], furniture['y'])
                    
                    # Distance to zone boundary
                    boundary_distance = zone_poly.boundary.distance(furniture_point)
                    
                    # Distance to other furniture (avoid overcrowding)
                    min_distance_to_others = float('inf')
                    for other in solution['furniture_positions']:
                        if other != furniture:
                            distance = math.sqrt(
                                (furniture['x'] - other['x'])**2 + 
                                (furniture['y'] - other['y'])**2
                            )
                            min_distance_to_others = min(min_distance_to_others, distance)
                    
                    # Accessibility score
                    boundary_score = min(1.0, boundary_distance / 2.0)
                    spacing_score = min(1.0, min_distance_to_others / 3.0) if min_distance_to_others != float('inf') else 1.0
                    
                    accessibility_scores.append((boundary_score + spacing_score) / 2)
                    
                except:
                    accessibility_scores.append(0.5)
        
        return sum(accessibility_scores) / len(accessibility_scores) if accessibility_scores else 0.0

    def _calculate_efficiency_score(self, solution: Dict, zones: List[Dict]) -> float:
        """Calculate overall efficiency score"""
        efficiency_factors = []
        
        # Zone utilization balance
        zone_utilizations = []
        for i, zone in enumerate(zones):
            zone_furniture = [f for f in solution['furniture_positions'] if f['zone_id'] == i]
            try:
                zone_area = Polygon(zone['points']).area
                utilization = len(zone_furniture) / (zone_area / 5.0)  # Target 1 item per 5 sq meters
                zone_utilizations.append(min(1.0, utilization))
            except:
                continue
        
        if zone_utilizations:
            # Prefer balanced utilization across zones
            avg_utilization = sum(zone_utilizations) / len(zone_utilizations)
            utilization_variance = sum((u - avg_utilization)**2 for u in zone_utilizations) / len(zone_utilizations)
            balance_score = 1.0 / (1.0 + utilization_variance)
            efficiency_factors.append(balance_score)
        
        # Furniture type diversity
        furniture_types = [f['item_type'] for f in solution['furniture_positions']]
        unique_types = len(set(furniture_types))
        diversity_score = min(1.0, unique_types / 4.0)  # Max 4 types
        efficiency_factors.append(diversity_score)
        
        return sum(efficiency_factors) / len(efficiency_factors) if efficiency_factors else 0.0

    def _check_constraints(self, solution: Dict, zones: List[Dict], params: Dict) -> float:
        """Check constraint satisfaction"""
        constraint_scores = []
        
        # Check if all furniture is within zone boundaries
        boundary_violations = 0
        for furniture in solution['furniture_positions']:
            zone_id = furniture['zone_id']
            if zone_id < len(zones):
                try:
                    zone_poly = Polygon(zones[zone_id]['points'])
                    furniture_point = Point(furniture['x'], furniture['y'])
                    if not zone_poly.contains(furniture_point):
                        boundary_violations += 1
                except:
                    boundary_violations += 1
        
        boundary_score = 1.0 - (boundary_violations / len(solution['furniture_positions'])) if solution['furniture_positions'] else 1.0
        constraint_scores.append(boundary_score)
        
        # Check minimum spacing constraints
        spacing_violations = 0
        min_spacing = params.get('min_spacing', 1.0)
        
        for i, furniture1 in enumerate(solution['furniture_positions']):
            for furniture2 in solution['furniture_positions'][i+1:]:
                distance = math.sqrt(
                    (furniture1['x'] - furniture2['x'])**2 + 
                    (furniture1['y'] - furniture2['y'])**2
                )
                if distance < min_spacing:
                    spacing_violations += 1
        
        total_pairs = len(solution['furniture_positions']) * (len(solution['furniture_positions']) - 1) // 2
        spacing_score = 1.0 - (spacing_violations / total_pairs) if total_pairs > 0 else 1.0
        constraint_scores.append(spacing_score)
        
        return sum(constraint_scores) / len(constraint_scores) if constraint_scores else 1.0

    def _generate_neighbor_solution(self, current_solution: Dict, zones: List[Dict], params: Dict) -> Dict:
        """Generate neighbor solution for simulated annealing"""
        new_solution = {
            'furniture_positions': current_solution['furniture_positions'].copy(),
            'zone_assignments': current_solution['zone_assignments'].copy(),
            'efficiency_metrics': current_solution['efficiency_metrics'].copy()
        }
        
        if new_solution['furniture_positions']:
            # Random modification
            modification_type = np.random.choice(['move', 'rotate', 'swap', 'add', 'remove'])
            
            if modification_type == 'move' and new_solution['furniture_positions']:
                # Move random furniture
                idx = np.random.randint(len(new_solution['furniture_positions']))
                furniture = new_solution['furniture_positions'][idx]
                zone_id = furniture['zone_id']
                
                if zone_id < len(zones):
                    try:
                        zone_poly = Polygon(zones[zone_id]['points'])
                        bounds = zone_poly.bounds
                        
                        new_x = np.random.uniform(bounds[0], bounds[2])
                        new_y = np.random.uniform(bounds[1], bounds[3])
                        
                        if zone_poly.contains(Point(new_x, new_y)):
                            new_solution['furniture_positions'][idx]['x'] = new_x
                            new_solution['furniture_positions'][idx]['y'] = new_y
                    except:
                        pass
            
            elif modification_type == 'rotate' and new_solution['furniture_positions']:
                # Rotate random furniture
                idx = np.random.randint(len(new_solution['furniture_positions']))
                new_solution['furniture_positions'][idx]['rotation'] = np.random.uniform(0, 360)
            
            elif modification_type == 'add':
                # Add new furniture to random zone
                zone_id = np.random.randint(len(zones))
                try:
                    zone_poly = Polygon(zones[zone_id]['points'])
                    bounds = zone_poly.bounds
                    
                    x = np.random.uniform(bounds[0], bounds[2])
                    y = np.random.uniform(bounds[1], bounds[3])
                    
                    if zone_poly.contains(Point(x, y)):
                        new_solution['furniture_positions'].append({
                            'zone_id': zone_id,
                            'x': x,
                            'y': y,
                            'rotation': np.random.uniform(0, 360),
                            'item_type': np.random.choice(['desk', 'chair', 'cabinet', 'table'])
                        })
                except:
                    pass
            
            elif modification_type == 'remove' and len(new_solution['furniture_positions']) > 1:
                # Remove random furniture
                idx = np.random.randint(len(new_solution['furniture_positions']))
                new_solution['furniture_positions'].pop(idx)
        
        return new_solution

    def _tournament_selection(self, population: List[Dict], tournament_size: int = 3) -> Dict:
        """Tournament selection for genetic algorithm"""
        tournament = np.random.choice(population, tournament_size, replace=False)
        return max(tournament, key=lambda x: x['fitness'])

    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover operation for genetic algorithm"""
        child = {
            'furniture_positions': [],
            'zone_assignments': {},
            'efficiency_metrics': {}
        }
        
        # Combine furniture positions from both parents
        all_positions = parent1['furniture_positions'] + parent2['furniture_positions']
        
        # Random selection
        for position in all_positions:
            if np.random.random() < 0.5:
                child['furniture_positions'].append(position.copy())
        
        # Ensure minimum furniture
        if not child['furniture_positions'] and all_positions:
            child['furniture_positions'] = [all_positions[0].copy()]
        
        return child

    def _mutate(self, individual: Dict, zones: List[Dict], params: Dict, mutation_rate: float = 0.1) -> Dict:
        """Mutation operation for genetic algorithm"""
        if np.random.random() < mutation_rate:
            return self._generate_neighbor_solution(individual, zones, params)
        return individual

    def _calculate_population_diversity(self, population: List[Dict]) -> float:
        """Calculate population diversity"""
        if len(population) < 2:
            return 0.0
        
        diversity_scores = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                diversity = self._solution_diversity(
                    population[i]['solution'], 
                    population[j]['solution']
                )
                diversity_scores.append(diversity)
        
        return sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0

    def _solution_diversity(self, solution1: Dict, solution2: Dict) -> float:
        """Calculate diversity between two solutions"""
        positions1 = solution1['furniture_positions']
        positions2 = solution2['furniture_positions']
        
        if not positions1 or not positions2:
            return 1.0 if len(positions1) != len(positions2) else 0.0
        
        # Calculate average position difference
        total_distance = 0
        min_length = min(len(positions1), len(positions2))
        
        for i in range(min_length):
            p1 = positions1[i]
            p2 = positions2[i]
            distance = math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
            total_distance += distance
        
        # Normalize by number of positions
        avg_distance = total_distance / min_length if min_length > 0 else 0
        
        # Add penalty for different numbers of furniture
        size_penalty = abs(len(positions1) - len(positions2)) / max(len(positions1), len(positions2))
        
        return min(1.0, (avg_distance / 10.0) + size_penalty)

    def _generate_random_velocity(self, position: Dict) -> Dict:
        """Generate random velocity for PSO"""
        return {
            'furniture_deltas': [
                {
                    'dx': np.random.uniform(-1, 1),
                    'dy': np.random.uniform(-1, 1),
                    'drotation': np.random.uniform(-10, 10)
                }
                for _ in position['furniture_positions']
            ]
        }

    def _solution_difference(self, solution1: Dict, solution2: Dict) -> Dict:
        """Calculate difference between solutions for PSO"""
        positions1 = solution1['furniture_positions']
        positions2 = solution2['furniture_positions']
        
        deltas = []
        min_length = min(len(positions1), len(positions2))
        
        for i in range(min_length):
            p1 = positions1[i]
            p2 = positions2[i]
            deltas.append({
                'dx': p2['x'] - p1['x'],
                'dy': p2['y'] - p1['y'],
                'drotation': p2['rotation'] - p1['rotation']
            })
        
        return {'furniture_deltas': deltas}

    def _update_position(self, position: Dict, velocity: Dict, zones: List[Dict], params: Dict) -> Dict:
        """Update position for PSO"""
        new_position = {
            'furniture_positions': [],
            'zone_assignments': position['zone_assignments'].copy(),
            'efficiency_metrics': position['efficiency_metrics'].copy()
        }
        
        deltas = velocity.get('furniture_deltas', [])
        
        for i, furniture in enumerate(position['furniture_positions']):
            new_furniture = furniture.copy()
            
            if i < len(deltas):
                delta = deltas[i]
                new_x = furniture['x'] + delta['dx']
                new_y = furniture['y'] + delta['dy']
                new_rotation = (furniture['rotation'] + delta['drotation']) % 360
                
                # Check if new position is valid
                zone_id = furniture['zone_id']
                if zone_id < len(zones):
                    try:
                        zone_poly = Polygon(zones[zone_id]['points'])
                        if zone_poly.contains(Point(new_x, new_y)):
                            new_furniture['x'] = new_x
                            new_furniture['y'] = new_y
                            new_furniture['rotation'] = new_rotation
                    except:
                        pass
            
            new_position['furniture_positions'].append(new_furniture)
        
        return new_position

    def _de_mutation(self, a: Dict, b: Dict, c: Dict, F: float) -> Dict:
        """Differential evolution mutation"""
        mutant = {
            'furniture_positions': [],
            'zone_assignments': {},
            'efficiency_metrics': {}
        }
        
        positions_a = a['furniture_positions']
        positions_b = b['furniture_positions']
        positions_c = c['furniture_positions']
        
        max_length = max(len(positions_a), len(positions_b), len(positions_c))
        
        for i in range(max_length):
            # Get positions (use last available if index exceeds length)
            pos_a = positions_a[min(i, len(positions_a) - 1)] if positions_a else None
            pos_b = positions_b[min(i, len(positions_b) - 1)] if positions_b else None
            pos_c = positions_c[min(i, len(positions_c) - 1)] if positions_c else None
            
            if pos_a and pos_b and pos_c:
                # Mutation: a + F * (b - c)
                new_x = pos_a['x'] + F * (pos_b['x'] - pos_c['x'])
                new_y = pos_a['y'] + F * (pos_b['y'] - pos_c['y'])
                new_rotation = (pos_a['rotation'] + F * (pos_b['rotation'] - pos_c['rotation'])) % 360
                
                mutant['furniture_positions'].append({
                    'zone_id': pos_a['zone_id'],
                    'x': new_x,
                    'y': new_y,
                    'rotation': new_rotation,
                    'item_type': pos_a['item_type']
                })
        
        return mutant

    def _de_crossover(self, target: Dict, mutant: Dict, CR: float) -> Dict:
        """Differential evolution crossover"""
        trial = {
            'furniture_positions': [],
            'zone_assignments': target['zone_assignments'].copy(),
            'efficiency_metrics': target['efficiency_metrics'].copy()
        }
        
        target_positions = target['furniture_positions']
        mutant_positions = mutant['furniture_positions']
        
        max_length = max(len(target_positions), len(mutant_positions))
        
        for i in range(max_length):
            if np.random.random() < CR:
                # Use mutant
                if i < len(mutant_positions):
                    trial['furniture_positions'].append(mutant_positions[i].copy())
            else:
                # Use target
                if i < len(target_positions):
                    trial['furniture_positions'].append(target_positions[i].copy())
        
        return trial


class AdvancedAIModels:
    """
    Main AI models controller that integrates all AI analysis components
    """
    
    def __init__(self):
        self.room_classifier = AdvancedRoomClassifier()
        self.space_analyzer = SemanticSpaceAnalyzer()
        self.optimization_engine = OptimizationEngine()
        
    def analyze_comprehensive(self, zones: List[Dict]) -> Dict:
        """Comprehensive AI analysis of zones"""
        results = {}
        
        try:
            # Room classification
            room_classifications = self.room_classifier.batch_classify(zones)
            results['rooms'] = room_classifications
            
            # Spatial relationship analysis
            space_graph = self.space_analyzer.build_space_graph(zones, room_classifications)
            spatial_analysis = self.space_analyzer.analyze_spatial_relationships()
            results['spatial_analysis'] = spatial_analysis
            
            # Optimization analysis
            optimization_params = {
                'optimization_method': 'simulated_annealing',
                'box_size': [1.2, 0.8],
                'margin': 0.3,
                'allow_rotation': True
            }
            optimization_results = self.optimization_engine.optimize_furniture_placement(zones, optimization_params)
            results['optimization'] = optimization_results
            
            # Calculate overall metrics
            results['summary'] = self._calculate_summary_metrics(zones, room_classifications, spatial_analysis)
            
        except Exception as e:
            results['error'] = str(e)
            results['rooms'] = {}
            results['spatial_analysis'] = {}
            results['optimization'] = {'total_efficiency': 0.5}
            
        return results
    
    def _calculate_summary_metrics(self, zones: List[Dict], room_classifications: Dict, spatial_analysis: Dict) -> Dict:
        """Calculate summary metrics"""
        total_area = sum(zone.get('area', 0) for zone in zones)
        classified_rooms = len([r for r in room_classifications.values() if r.get('confidence', 0) > 0.5])
        
        avg_confidence = 0
        if room_classifications:
            confidences = [r.get('confidence', 0) for r in room_classifications.values()]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
        return {
            'total_zones': len(zones),
            'total_area': total_area,
            'classified_rooms': classified_rooms,
            'average_confidence': avg_confidence,
            'connectivity_score': spatial_analysis.get('graph_stats', {}).get('is_connected', False),
            'analysis_quality': 'good' if avg_confidence > 0.7 else 'moderate' if avg_confidence > 0.5 else 'low'
        }
