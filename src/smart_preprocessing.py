
"""
Smart Preprocessing System for Faster File Analysis
"""
import streamlit as st
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import cv2
import json
from PIL import Image
import io
import hashlib
from .caching_system import memory_cache, session_cache

class SmartPreprocessor:
    """Intelligent preprocessing for files before analysis"""
    
    def __init__(self):
        self.file_signatures = {
            'pdf': [b'%PDF'],
            'dwg': [b'AC10', b'AC12', b'AC14', b'AC15', b'AC18', b'AC21', b'AC24', b'AC27'],
            'dxf': [b'0\r\nSECTION', b'999\r\nDXF'],
            'image': [b'\xff\xd8\xff', b'\x89PNG', b'GIF8']
        }
        
    @memory_cache(ttl=1800)  # 30 minutes cache
    def quick_file_analysis(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Quick analysis of file characteristics before full parsing"""
        file_hash = hashlib.md5(file_bytes).hexdigest()
        file_size = len(file_bytes)
        file_type = self._detect_file_type(file_bytes, filename)
        
        analysis = {
            'file_hash': file_hash,
            'file_size': file_size,
            'file_type': file_type,
            'estimated_complexity': self._estimate_complexity(file_bytes, file_type),
            'processing_strategy': self._determine_processing_strategy(file_size, file_type),
            'can_fast_track': self._can_fast_track(file_size, file_type)
        }
        
        return analysis
    
    def _detect_file_type(self, file_bytes: bytes, filename: str) -> str:
        """Detect file type from content and extension"""
        # Check file signatures
        for file_type, signatures in self.file_signatures.items():
            for signature in signatures:
                if file_bytes.startswith(signature):
                    return file_type
        
        # Fallback to extension
        extension = filename.lower().split('.')[-1]
        type_mapping = {
            'pdf': 'pdf',
            'dwg': 'dwg',
            'dxf': 'dxf',
            'jpg': 'image',
            'jpeg': 'image',
            'png': 'image',
            'bmp': 'image'
        }
        
        return type_mapping.get(extension, 'unknown')
    
    def _estimate_complexity(self, file_bytes: bytes, file_type: str) -> str:
        """Estimate file complexity for processing time prediction"""
        file_size = len(file_bytes)
        
        if file_type == 'pdf':
            # Count PDF objects (rough estimation)
            obj_count = file_bytes.count(b'obj')
            if obj_count < 50:
                return 'low'
            elif obj_count < 200:
                return 'medium'
            else:
                return 'high'
                
        elif file_type in ['dwg', 'dxf']:
            # Estimate based on file size
            if file_size < 100 * 1024:  # < 100KB
                return 'low'
            elif file_size < 1024 * 1024:  # < 1MB
                return 'medium'
            else:
                return 'high'
                
        elif file_type == 'image':
            if file_size < 500 * 1024:  # < 500KB
                return 'low'
            elif file_size < 2 * 1024 * 1024:  # < 2MB
                return 'medium'
            else:
                return 'high'
        
        return 'medium'
    
    def _determine_processing_strategy(self, file_size: int, file_type: str) -> str:
        """Determine optimal processing strategy"""
        if file_size < 100 * 1024:  # < 100KB
            return 'sequential'
        elif file_size < 1024 * 1024:  # < 1MB
            return 'parallel_light'
        else:
            return 'parallel_heavy'
    
    def _can_fast_track(self, file_size: int, file_type: str) -> bool:
        """Determine if file can use fast-track processing"""
        if file_type == 'image' and file_size < 1024 * 1024:
            return True
        elif file_type in ['dwg', 'dxf'] and file_size < 500 * 1024:
            return True
        elif file_type == 'pdf' and file_size < 2 * 1024 * 1024:
            return True
        return False
    
    @session_cache(ttl=900)  # 15 minutes cache
    def preprocess_pdf_fast(self, file_bytes: bytes) -> Dict[str, Any]:
        """Fast preprocessing for PDF files"""
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            
            # Quick analysis
            page_count = len(doc)
            text_content = ""
            has_images = False
            
            # Analyze first few pages for speed
            sample_pages = min(3, page_count)
            
            for page_num in range(sample_pages):
                page = doc[page_num]
                text_content += page.get_text()[:1000]  # First 1000 chars
                
                # Check for images
                if not has_images:
                    images = page.get_images()
                    has_images = len(images) > 0
            
            doc.close()
            
            return {
                'page_count': page_count,
                'has_text': len(text_content.strip()) > 0,
                'has_images': has_images,
                'sample_text': text_content[:500],
                'processing_recommendation': 'vector' if not has_images else 'hybrid'
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'processing_recommendation': 'fallback'
            }
    
    @session_cache(ttl=1200)  # 20 minutes cache
    def preprocess_dwg_fast(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Fast preprocessing for DWG/DXF files"""
        try:
            import ezdxf
            import tempfile
            import os
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.dxf', delete=False) as temp_file:
                temp_file.write(file_bytes)
                temp_path = temp_file.name
            
            try:
                doc = ezdxf.readfile(temp_path)
                msp = doc.modelspace()
                
                # Quick entity count
                entity_count = len(list(msp))
                layer_count = len(list(doc.layers))
                
                # Sample entity types
                entity_types = set()
                sample_count = 0
                for entity in msp:
                    entity_types.add(entity.dxftype())
                    sample_count += 1
                    if sample_count >= 100:  # Sample first 100
                        break
                
                return {
                    'entity_count': entity_count,
                    'layer_count': layer_count,
                    'entity_types': list(entity_types),
                    'complexity': 'low' if entity_count < 1000 else 'medium' if entity_count < 5000 else 'high',
                    'processing_recommendation': 'batch' if entity_count > 1000 else 'sequential'
                }
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            return {
                'error': str(e),
                'processing_recommendation': 'fallback'
            }
    
    def optimize_zones_for_processing(self, zones: List[Dict]) -> List[Dict]:
        """Optimize zone data structure for faster processing"""
        optimized_zones = []
        
        for zone in zones:
            optimized_zone = {
                'zone_id': zone.get('zone_id', ''),
                'points': self._optimize_points(zone.get('points', [])),
                'area': zone.get('area', 0),
                'zone_type': zone.get('zone_type', 'Unknown'),
                'layer': zone.get('layer', ''),
                'bounds': zone.get('bounds', {}),
                'processing_priority': self._calculate_priority(zone)
            }
            optimized_zones.append(optimized_zone)
        
        # Sort by processing priority
        optimized_zones.sort(key=lambda x: x['processing_priority'], reverse=True)
        
        return optimized_zones
    
    def _optimize_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Optimize point data for faster processing"""
        if len(points) <= 4:
            return points
        
        # Simplify complex polygons using Douglas-Peucker algorithm
        if len(points) > 20:
            # Convert to numpy array
            points_array = np.array(points)
            
            # Simple decimation for very complex polygons
            step = max(1, len(points) // 20)
            simplified = points_array[::step].tolist()
            
            # Ensure polygon is closed
            if simplified[0] != simplified[-1]:
                simplified.append(simplified[0])
            
            return simplified
        
        return points
    
    def _calculate_priority(self, zone: Dict) -> float:
        """Calculate processing priority for zone"""
        priority = 0.0
        
        # Area-based priority
        area = zone.get('area', 0)
        if area > 50:
            priority += 3.0
        elif area > 20:
            priority += 2.0
        elif area > 5:
            priority += 1.0
        
        # Type-based priority
        zone_type = zone.get('zone_type', '').lower()
        type_priorities = {
            'office': 2.0,
            'meeting room': 2.0,
            'conference room': 1.5,
            'corridor': 0.5,
            'storage': 0.3
        }
        
        for type_name, type_priority in type_priorities.items():
            if type_name in zone_type:
                priority += type_priority
                break
        
        # Complexity-based priority
        points = zone.get('points', [])
        if len(points) > 10:
            priority += 1.0
        elif len(points) > 6:
            priority += 0.5
        
        return priority
    
    def create_processing_plan(self, zones: List[Dict], analysis_type: str) -> Dict[str, Any]:
        """Create optimized processing plan"""
        zone_count = len(zones)
        total_area = sum(zone.get('area', 0) for zone in zones)
        
        # Determine batch size based on complexity
        if zone_count <= 10:
            batch_size = zone_count
            strategy = 'sequential'
        elif zone_count <= 50:
            batch_size = max(5, zone_count // 4)
            strategy = 'parallel_light'
        else:
            batch_size = max(10, zone_count // 8)
            strategy = 'parallel_heavy'
        
        # Estimated processing time
        estimated_time = self._estimate_processing_time(zone_count, total_area, analysis_type)
        
        return {
            'strategy': strategy,
            'batch_size': batch_size,
            'estimated_time': estimated_time,
            'zone_count': zone_count,
            'total_area': total_area,
            'recommended_caching': zone_count > 20,
            'use_parallel': zone_count > 10
        }
    
    def _estimate_processing_time(self, zone_count: int, total_area: float, analysis_type: str) -> float:
        """Estimate processing time in seconds"""
        base_time_per_zone = {
            'basic': 0.1,
            'advanced': 0.3,
            'ai_analysis': 0.5,
            'optimization': 1.0
        }
        
        base_time = base_time_per_zone.get(analysis_type, 0.3)
        
        # Scale with zone count
        total_time = zone_count * base_time
        
        # Adjust for area complexity
        if total_area > 1000:
            total_time *= 1.5
        elif total_area > 500:
            total_time *= 1.2
        
        # Parallel processing reduction
        if zone_count > 10:
            parallel_factor = min(4, zone_count // 10)
            total_time /= parallel_factor
        
        return max(1.0, total_time)

# Global preprocessor instance
smart_preprocessor = SmartPreprocessor()
