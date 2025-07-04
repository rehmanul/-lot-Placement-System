
"""
Parallel Processing System for Enhanced Performance
"""
import streamlit as st
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import asyncio
import time
from typing import List, Dict, Any, Callable, Optional
import queue
import numpy as np
from functools import partial

class ParallelProcessor:
    """Advanced parallel processing for zone analysis and optimization"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(8, mp.cpu_count())
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=min(4, mp.cpu_count()))
        
    def process_zones_parallel(self, zones: List[Dict], analysis_func: Callable) -> List[Dict]:
        """Process zones in parallel using threading"""
        if len(zones) <= 4:
            # Use sequential processing for small datasets
            return [analysis_func(zone) for zone in zones]
        
        # Show progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Batch processing
        batch_size = max(1, len(zones) // self.max_workers)
        batches = [zones[i:i + batch_size] for i in range(0, len(zones), batch_size)]
        
        results = []
        completed_batches = 0
        
        def process_batch(batch):
            return [analysis_func(zone) for zone in batch]
        
        # Submit all batches
        futures = [self.thread_pool.submit(process_batch, batch) for batch in batches]
        
        # Collect results
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                results.extend(batch_results)
                completed_batches += 1
                
                # Update progress
                progress = completed_batches / len(batches)
                progress_bar.progress(progress)
                status_text.text(f"Processing batch {completed_batches}/{len(batches)}")
                
            except Exception as e:
                st.error(f"Error processing batch: {str(e)}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        return results
    
    def process_zones_async(self, zones: List[Dict], analysis_func: Callable) -> List[Dict]:
        """Process zones asynchronously with real-time updates"""
        async def async_process_zone(zone, semaphore):
            async with semaphore:
                # Simulate async processing
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, analysis_func, zone)
                return result
        
        async def process_all_zones():
            semaphore = asyncio.Semaphore(self.max_workers)
            tasks = [async_process_zone(zone, semaphore) for zone in zones]
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, task in enumerate(asyncio.as_completed(tasks)):
                result = await task
                results.append(result)
                
                # Update progress
                progress = (i + 1) / len(tasks)
                progress_bar.progress(progress)
                status_text.text(f"Processed {i + 1}/{len(tasks)} zones")
            
            progress_bar.empty()
            status_text.empty()
            return results
        
        # Run async processing
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(process_all_zones())
        except Exception as e:
            st.error(f"Async processing error: {str(e)}")
            return [analysis_func(zone) for zone in zones]
        finally:
            loop.close()
    
    def batch_optimize_furniture(self, zones: List[Dict], optimization_params: Dict) -> Dict:
        """Optimize furniture placement using parallel processing"""
        from .optimization import OptimizationEngine
        
        optimizer = OptimizationEngine()
        
        if len(zones) <= 2:
            return optimizer.optimize_furniture_placement(zones, optimization_params)
        
        # Split zones into batches
        batch_size = max(1, len(zones) // 4)
        zone_batches = [zones[i:i + batch_size] for i in range(0, len(zones), batch_size)]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def optimize_batch(batch, params):
            return optimizer.optimize_furniture_placement(batch, params)
        
        batch_results = []
        
        # Process batches in parallel
        futures = [
            self.thread_pool.submit(optimize_batch, batch, optimization_params)
            for batch in zone_batches
        ]
        
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                batch_results.append(result)
                
                progress = (i + 1) / len(futures)
                progress_bar.progress(progress)
                status_text.text(f"Optimizing batch {i + 1}/{len(futures)}")
                
            except Exception as e:
                st.error(f"Optimization error: {str(e)}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        # Combine batch results
        combined_result = self._combine_optimization_results(batch_results)
        return combined_result
    
    def parallel_room_classification(self, zones: List[Dict]) -> Dict:
        """Classify rooms in parallel"""
        from .advanced_ai_models import AdvancedRoomClassifier
        
        classifier = AdvancedRoomClassifier()
        
        # Split zones for parallel processing
        zone_chunks = np.array_split(zones, min(self.max_workers, len(zones)))
        
        def classify_chunk(chunk):
            chunk_dict = {i: zone for i, zone in enumerate(chunk)}
            return classifier.batch_classify(list(chunk_dict.values()))
        
        progress_bar = st.progress(0)
        futures = [self.thread_pool.submit(classify_chunk, chunk) for chunk in zone_chunks]
        
        all_results = {}
        for i, future in enumerate(as_completed(futures)):
            try:
                chunk_results = future.result()
                # Adjust indices for global zone numbering
                start_idx = i * len(zone_chunks[i])
                for local_idx, result in chunk_results.items():
                    global_idx = start_idx + local_idx
                    all_results[global_idx] = result
                
                progress = (i + 1) / len(futures)
                progress_bar.progress(progress)
                
            except Exception as e:
                st.error(f"Classification error: {str(e)}")
                continue
        
        progress_bar.empty()
        return all_results
    
    def real_time_processing_queue(self, zones: List[Dict]) -> queue.Queue:
        """Set up real-time processing queue for live updates"""
        result_queue = queue.Queue()
        
        def process_zone_realtime(zone, zone_index):
            try:
                # Simulate processing with multiple steps
                steps = ['parsing', 'analyzing', 'optimizing', 'finalizing']
                
                for step in steps:
                    time.sleep(0.1)  # Simulate processing time
                    result_queue.put({
                        'zone_index': zone_index,
                        'step': step,
                        'status': 'processing',
                        'progress': (steps.index(step) + 1) / len(steps)
                    })
                
                # Final result
                result_queue.put({
                    'zone_index': zone_index,
                    'step': 'complete',
                    'status': 'completed',
                    'result': zone,
                    'progress': 1.0
                })
                
            except Exception as e:
                result_queue.put({
                    'zone_index': zone_index,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Start processing threads
        for i, zone in enumerate(zones):
            thread = threading.Thread(target=process_zone_realtime, args=(zone, i))
            thread.daemon = True
            thread.start()
        
        return result_queue
    
    def _combine_optimization_results(self, batch_results: List[Dict]) -> Dict:
        """Combine optimization results from multiple batches"""
        combined = {
            'total_efficiency': 0.0,
            'placements': {},
            'optimization_details': {},
            'total_boxes': 0,
            'algorithm_used': 'Parallel Batch Optimization'
        }
        
        if not batch_results:
            return combined
        
        # Calculate weighted average efficiency
        total_weight = 0
        weighted_efficiency = 0
        
        for result in batch_results:
            efficiency = result.get('total_efficiency', 0)
            boxes = result.get('total_boxes', 0)
            weight = max(1, boxes)  # Use number of boxes as weight
            
            weighted_efficiency += efficiency * weight
            total_weight += weight
            combined['total_boxes'] += boxes
            
            # Merge placements
            placements = result.get('placements', {})
            for zone_id, placement in placements.items():
                combined['placements'][f"batch_{len(combined['placements'])}_{zone_id}"] = placement
        
        combined['total_efficiency'] = weighted_efficiency / max(1, total_weight)
        combined['optimization_details'] = {
            'batch_count': len(batch_results),
            'combined_efficiency': combined['total_efficiency'],
            'parallel_processing': True
        }
        
        return combined
    
    def cleanup(self):
        """Clean up thread and process pools"""
        try:
            self.thread_pool.shutdown(wait=True)
            self.process_pool.shutdown(wait=True)
        except Exception as e:
            print(f"Cleanup error: {str(e)}")

# Global parallel processor instance
parallel_processor = ParallelProcessor()
