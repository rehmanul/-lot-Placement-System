
"""
Advanced Caching System for Performance Optimization
"""
import streamlit as st
import hashlib
import time
import pickle
import json
from typing import Any, Dict, List, Optional, Callable
from functools import wraps
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio

class AdvancedCacheManager:
    """Advanced caching system with multiple cache levels"""
    
    def __init__(self):
        self.memory_cache = {}
        self.session_cache = {}
        self.file_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'total_requests': 0
        }
        
    def generate_cache_key(self, *args, **kwargs) -> str:
        """Generate unique cache key from arguments"""
        content = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(content.encode()).hexdigest()
    
    def memory_cache_decorator(self, ttl: int = 300):
        """Memory cache decorator with TTL"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                cache_key = f"{func.__name__}_{self.generate_cache_key(*args, **kwargs)}"
                
                # Check memory cache
                if cache_key in self.memory_cache:
                    cached_data = self.memory_cache[cache_key]
                    if time.time() - cached_data['timestamp'] < ttl:
                        self.cache_stats['hits'] += 1
                        self.cache_stats['total_requests'] += 1
                        return cached_data['result']
                    else:
                        # Remove expired cache
                        del self.memory_cache[cache_key]
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                self.memory_cache[cache_key] = {
                    'result': result,
                    'timestamp': time.time()
                }
                
                self.cache_stats['misses'] += 1
                self.cache_stats['total_requests'] += 1
                return result
            
            return wrapper
        return decorator
    
    def session_cache_decorator(self, ttl: int = 600):
        """Session-based cache decorator"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if 'advanced_cache' not in st.session_state:
                    st.session_state.advanced_cache = {}
                
                cache_key = f"{func.__name__}_{self.generate_cache_key(*args, **kwargs)}"
                
                # Check session cache
                if cache_key in st.session_state.advanced_cache:
                    cached_data = st.session_state.advanced_cache[cache_key]
                    if time.time() - cached_data['timestamp'] < ttl:
                        return cached_data['result']
                    else:
                        del st.session_state.advanced_cache[cache_key]
                
                # Execute and cache
                result = func(*args, **kwargs)
                st.session_state.advanced_cache[cache_key] = {
                    'result': result,
                    'timestamp': time.time()
                }
                
                return result
            
            return wrapper
        return decorator
    
    def preload_cache(self, data_sources: List[Dict]):
        """Preload commonly used data into cache"""
        for source in data_sources:
            cache_key = source['key']
            if source['type'] == 'room_templates':
                self.memory_cache[cache_key] = {
                    'result': self._generate_room_templates(),
                    'timestamp': time.time()
                }
            elif source['type'] == 'furniture_catalog':
                self.memory_cache[cache_key] = {
                    'result': self._generate_furniture_catalog(),
                    'timestamp': time.time()
                }
    
    def _generate_room_templates(self) -> Dict:
        """Generate common room templates"""
        return {
            'Office': {
                'min_area': 9, 'max_area': 30, 'aspect_ratio': (1.0, 2.0),
                'furniture': ['desk', 'chair', 'storage']
            },
            'Meeting Room': {
                'min_area': 15, 'max_area': 40, 'aspect_ratio': (1.0, 1.8),
                'furniture': ['table', 'chairs', 'projector']
            },
            'Kitchen': {
                'min_area': 8, 'max_area': 25, 'aspect_ratio': (1.0, 2.5),
                'furniture': ['counter', 'sink', 'appliances']
            }
        }
    
    def _generate_furniture_catalog(self) -> Dict:
        """Generate furniture catalog"""
        return {
            'desk': {'width': 1.5, 'depth': 0.8, 'category': 'work'},
            'chair': {'width': 0.6, 'depth': 0.6, 'category': 'seating'},
            'table': {'width': 1.2, 'depth': 0.8, 'category': 'work'},
            'sofa': {'width': 2.0, 'depth': 0.9, 'category': 'seating'},
            'cabinet': {'width': 0.8, 'depth': 0.4, 'category': 'storage'}
        }
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        hit_rate = (self.cache_stats['hits'] / max(1, self.cache_stats['total_requests'])) * 100
        return {
            'hit_rate': hit_rate,
            'total_hits': self.cache_stats['hits'],
            'total_misses': self.cache_stats['misses'],
            'memory_cache_size': len(self.memory_cache),
            'session_cache_size': len(getattr(st.session_state, 'advanced_cache', {}))
        }
    
    def clear_expired_cache(self, ttl: int = 300):
        """Clear expired cache entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, data in self.memory_cache.items():
            if current_time - data['timestamp'] > ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        return len(expired_keys)

# Global cache manager instance
cache_manager = AdvancedCacheManager()

# Decorators for easy use
memory_cache = cache_manager.memory_cache_decorator
session_cache = cache_manager.session_cache_decorator
