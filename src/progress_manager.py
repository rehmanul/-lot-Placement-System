
"""
Advanced Progress Management System
"""
import streamlit as st
import time
import threading
from typing import Dict, List, Any, Optional, Callable
import queue
import json
from datetime import datetime

class ProgressManager:
    """Advanced progress tracking with real-time updates"""
    
    def __init__(self):
        self.active_tasks = {}
        self.task_history = []
        self.progress_queue = queue.Queue()
        
    def create_progress_tracker(self, task_id: str, total_steps: int, description: str = "") -> 'ProgressTracker':
        """Create a new progress tracker for a task"""
        tracker = ProgressTracker(task_id, total_steps, description, self)
        self.active_tasks[task_id] = tracker
        return tracker
    
    def display_progress_dashboard(self):
        """Display comprehensive progress dashboard"""
        if not self.active_tasks:
            return
        
        st.subheader("🔄 Processing Status")
        
        # Overall progress
        total_tasks = len(self.active_tasks)
        completed_tasks = sum(1 for task in self.active_tasks.values() if task.is_complete())
        
        overall_progress = completed_tasks / total_tasks if total_tasks > 0 else 0
        st.progress(overall_progress)
        st.write(f"Overall Progress: {completed_tasks}/{total_tasks} tasks completed")
        
        # Individual task progress
        for task_id, tracker in self.active_tasks.items():
            with st.expander(f"📋 {tracker.description or task_id}", expanded=True):
                tracker.display()
    
    def cleanup_completed_tasks(self):
        """Clean up completed tasks"""
        completed_tasks = [task_id for task_id, tracker in self.active_tasks.items() 
                          if tracker.is_complete()]
        
        for task_id in completed_tasks:
            tracker = self.active_tasks.pop(task_id)
            self.task_history.append({
                'task_id': task_id,
                'description': tracker.description,
                'completion_time': datetime.now(),
                'total_time': tracker.get_elapsed_time(),
                'status': 'completed'
            })

class ProgressTracker:
    """Individual task progress tracker"""
    
    def __init__(self, task_id: str, total_steps: int, description: str, manager: ProgressManager):
        self.task_id = task_id
        self.total_steps = total_steps
        self.description = description
        self.manager = manager
        self.current_step = 0
        self.start_time = time.time()
        self.current_status = "Initializing..."
        self.step_details = []
        self.error_messages = []
        
        # Streamlit components
        self.progress_bar = None
        self.status_text = None
        self.details_container = None
        
    def update(self, step: int = None, status: str = None, details: str = None):
        """Update progress tracker"""
        if step is not None:
            self.current_step = min(step, self.total_steps)
        
        if status is not None:
            self.current_status = status
        
        if details is not None:
            self.step_details.append({
                'timestamp': time.time(),
                'step': self.current_step,
                'details': details
            })
        
        # Update display if components exist
        if self.progress_bar is not None:
            progress = self.current_step / self.total_steps if self.total_steps > 0 else 0
            self.progress_bar.progress(progress)
        
        if self.status_text is not None:
            elapsed = self.get_elapsed_time()
            eta = self.estimate_time_remaining()
            status_msg = f"{self.current_status} | Elapsed: {elapsed:.1f}s"
            if eta > 0:
                status_msg += f" | ETA: {eta:.1f}s"
            self.status_text.text(status_msg)
    
    def increment(self, status: str = None, details: str = None):
        """Increment progress by one step"""
        self.update(self.current_step + 1, status, details)
    
    def add_error(self, error_msg: str):
        """Add error message"""
        self.error_messages.append({
            'timestamp': time.time(),
            'step': self.current_step,
            'error': error_msg
        })
    
    def is_complete(self) -> bool:
        """Check if task is complete"""
        return self.current_step >= self.total_steps
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        return time.time() - self.start_time
    
    def estimate_time_remaining(self) -> float:
        """Estimate remaining time based on current progress"""
        if self.current_step == 0:
            return 0
        
        elapsed = self.get_elapsed_time()
        progress_ratio = self.current_step / self.total_steps
        
        if progress_ratio > 0:
            estimated_total = elapsed / progress_ratio
            remaining = estimated_total - elapsed
            return max(0, remaining)
        
        return 0
    
    def display(self):
        """Display progress tracker in Streamlit"""
        progress = self.current_step / self.total_steps if self.total_steps > 0 else 0
        
        # Progress bar
        self.progress_bar = st.progress(progress)
        
        # Status text
        elapsed = self.get_elapsed_time()
        eta = self.estimate_time_remaining()
        status_msg = f"{self.current_status} | Step {self.current_step}/{self.total_steps}"
        status_msg += f" | Elapsed: {elapsed:.1f}s"
        if eta > 0:
            status_msg += f" | ETA: {eta:.1f}s"
        
        self.status_text = st.text(status_msg)
        
        # Details and errors
        if self.step_details or self.error_messages:
            with st.expander("📝 Details", expanded=False):
                if self.error_messages:
                    st.error("⚠️ Errors encountered:")
                    for error in self.error_messages[-3:]:  # Show last 3 errors
                        st.write(f"Step {error['step']}: {error['error']}")
                
                if self.step_details:
                    st.info("📋 Recent steps:")
                    for detail in self.step_details[-5:]:  # Show last 5 details
                        st.write(f"Step {detail['step']}: {detail['details']}")

class SmartProgressManager:
    """Intelligent progress management with predictions"""
    
    def __init__(self):
        self.progress_manager = ProgressManager()
        self.performance_history = {}
        
    def create_smart_tracker(self, task_type: str, data_size: int, description: str = "") -> ProgressTracker:
        """Create progress tracker with intelligent time estimation"""
        # Estimate steps based on task type and data size
        estimated_steps = self._estimate_steps(task_type, data_size)
        
        task_id = f"{task_type}_{int(time.time())}"
        tracker = self.progress_manager.create_progress_tracker(
            task_id, estimated_steps, description
        )
        
        return tracker
    
    def _estimate_steps(self, task_type: str, data_size: int) -> int:
        """Estimate number of steps based on task type and data"""
        base_steps = {
            'pdf_parsing': max(5, data_size // 100),
            'dwg_parsing': max(3, data_size // 50),
            'zone_analysis': max(10, data_size * 2),
            'ai_classification': max(8, data_size),
            'optimization': max(15, data_size * 3),
            'visualization': max(5, data_size // 20)
        }
        
        return base_steps.get(task_type, max(5, data_size))
    
    def track_file_processing(self, filename: str, file_size: int, file_type: str):
        """Track file processing with smart progress estimation"""
        description = f"Processing {filename} ({file_size // 1024}KB)"
        
        # Estimate complexity based on file size and type
        complexity_factor = {
            'pdf': 1.0,
            'dwg': 1.5,
            'dxf': 1.2,
            'image': 0.5
        }.get(file_type, 1.0)
        
        estimated_zones = max(1, (file_size // 10240) * complexity_factor)  # Rough estimation
        
        tracker = self.create_smart_tracker('file_processing', int(estimated_zones), description)
        
        return tracker
    
    def track_ai_analysis(self, zone_count: int):
        """Track AI analysis progress"""
        description = f"AI Analysis of {zone_count} zones"
        tracker = self.create_smart_tracker('ai_analysis', zone_count, description)
        
        return tracker
    
    def track_optimization(self, zone_count: int, optimization_type: str):
        """Track optimization progress"""
        description = f"{optimization_type} optimization for {zone_count} zones"
        
        # Optimization is typically more intensive
        complexity_multiplier = {
            'genetic_algorithm': 3.0,
            'simulated_annealing': 2.0,
            'particle_swarm': 2.5,
            'basic': 1.0
        }.get(optimization_type, 2.0)
        
        estimated_work = int(zone_count * complexity_multiplier)
        tracker = self.create_smart_tracker('optimization', estimated_work, description)
        
        return tracker
    
    def display_processing_summary(self):
        """Display processing performance summary"""
        with st.sidebar.expander("📊 Processing Summary"):
            if self.progress_manager.task_history:
                completed_tasks = len(self.progress_manager.task_history)
                total_time = sum(task['total_time'] for task in self.progress_manager.task_history)
                avg_time = total_time / completed_tasks if completed_tasks > 0 else 0
                
                st.metric("Completed Tasks", completed_tasks)
                st.metric("Total Processing Time", f"{total_time:.1f}s")
                st.metric("Average Task Time", f"{avg_time:.1f}s")
                
                # Recent tasks
                recent_tasks = self.progress_manager.task_history[-3:]
                st.write("**Recent Tasks:**")
                for task in recent_tasks:
                    st.write(f"• {task['description']}: {task['total_time']:.1f}s")

# Global progress manager
smart_progress = SmartProgressManager()
