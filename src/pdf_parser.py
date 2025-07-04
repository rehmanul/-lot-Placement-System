<<<<<<< HEAD
<<<<<<< HEAD

"""
Advanced PDF Parser for Architectural Plans
Supports text-based PDFs, scanned images, and hybrid documents
"""

import tempfile
import os
from typing import List, Dict, Any, Optional, Tuple
import io
import traceback

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class PDFParser:
    def __init__(self):
        self.supported_formats = ['.pdf']
        self.dpi = 300
        self.line_threshold = 50
        self.contour_area_threshold = 100

    def parse_file_simple(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Parse PDF file with multiple strategies
        """
        if not PYMUPDF_AVAILABLE:
            print("ERROR: PyMuPDF not available - cannot parse PDF files")
            return []

        file_ext = os.path.splitext(filename.lower())[1]
        if file_ext not in self.supported_formats:
            print(f"ERROR: File '{filename}' is not a supported PDF format")
            return []

        try:
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
                temp_file.write(file_bytes)
                temp_file_path = temp_file.name

            try:
                zones = []
                
                # Strategy 1: Extract vector graphics from PDF
                vector_zones = self._extract_vector_graphics(temp_file_path, filename)
                zones.extend(vector_zones)
                
                # Strategy 2: Computer vision on rasterized pages
                if CV2_AVAILABLE and len(zones) < 3:  # If vector extraction didn't work well
                    cv_zones = self._extract_with_computer_vision(temp_file_path, filename)
                    zones.extend(cv_zones)
                
                # Strategy 3: Text-based extraction
                text_zones = self._extract_text_based_geometry(temp_file_path, filename)
                zones.extend(text_zones)

                if not zones:
                    print(f"ERROR: No valid geometric data found in PDF '{filename}'")
                    return []

                print(f"SUCCESS: Parsed {len(zones)} zones from PDF '{filename}'")
                return zones

            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except Exception as e:
            print(f"ERROR: Failed to parse PDF '{filename}': {str(e)}")
            traceback.print_exc()
            return []

    def _extract_vector_graphics(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Extract vector graphics directly from PDF"""
        zones = []
        
        try:
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get page drawings (vector graphics)
                drawings = page.get_drawings()
                
                for i, drawing in enumerate(drawings):
                    try:
                        # Extract path items
                        if 'items' in drawing:
                            for item in drawing['items']:
                                if item[0] == 'l':  # Line
                                    start = item[1]
                                    end = item[2]
                                    # Could be part of a larger shape
                                elif item[0] == 're':  # Rectangle
                                    rect = item[1]
                                    points = [
                                        (rect.x0, rect.y0),
                                        (rect.x1, rect.y0),
                                        (rect.x1, rect.y1),
                                        (rect.x0, rect.y1)
                                    ]
                                    
                                    zone = {
                                        'zone_id': f"pdf_rect_{page_num}_{i}",
                                        'zone_type': 'Room',
                                        'points': points,
                                        'area': (rect.x1 - rect.x0) * (rect.y1 - rect.y0),
                                        'layer': f'Page_{page_num}',
                                        'entity_type': 'RECTANGLE',
                                        'page': page_num
                                    }
                                    zones.append(zone)
                                    
                                elif item[0] == 'c':  # Curve/Circle
                                    # Handle circular shapes
                                    pass
                    except Exception as e:
                        print(f"Warning: Error processing drawing item: {str(e)}")
                        continue
            
            doc.close()
            
        except Exception as e:
            print(f"Warning: Vector graphics extraction failed: {str(e)}")
        
        return zones

    def _extract_with_computer_vision(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Extract shapes using computer vision on rasterized PDF"""
        zones = []
        
        if not CV2_AVAILABLE:
            return zones
        
        try:
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Convert page to image
                mat = fitz.Matrix(self.dpi/72, self.dpi/72)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to OpenCV format
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    continue
                
                # Extract shapes from image
                page_zones = self._detect_shapes_in_image(img, page_num)
                zones.extend(page_zones)
            
            doc.close()
            
        except Exception as e:
            print(f"Warning: Computer vision extraction failed: {str(e)}")
        
        return zones

    def _detect_shapes_in_image(self, img: np.ndarray, page_num: int) -> List[Dict[str, Any]]:
        """Detect architectural shapes in image using computer vision"""
        zones = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using HoughLinesP
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                   minLineLength=self.line_threshold, maxLineGap=10)
            
            # Connect lines into polygons
            if lines is not None:
                polygons = self._connect_cv_lines_to_polygons(lines)
                
                for i, polygon in enumerate(polygons):
                    if len(polygon) >= 3:
                        zone = {
                            'zone_id': f"cv_polygon_{page_num}_{i}",
                            'zone_type': 'Room',
                            'points': polygon,
                            'area': self._calculate_polygon_area_coords(polygon),
                            'layer': f'CV_Page_{page_num}',
                            'entity_type': 'CV_POLYGON',
                            'page': page_num
                        }
                        zones.append(zone)
            
            # Detect contours for filled shapes
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > self.contour_area_threshold:
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) >= 3:
                        points = []
                        scale = 72.0 / self.dpi  # Convert pixels back to points
                        
                        for point in approx:
                            x, y = point[0]
                            points.append((x * scale, y * scale))
                        
                        zone = {
                            'zone_id': f"cv_contour_{page_num}_{i}",
                            'zone_type': 'Room',
                            'points': points,
                            'area': area * (scale ** 2),
                            'layer': f'CV_Contour_Page_{page_num}',
                            'entity_type': 'CV_CONTOUR',
                            'page': page_num
                        }
                        zones.append(zone)
            
        except Exception as e:
            print(f"Warning: Shape detection failed: {str(e)}")
        
        return zones

    def _connect_cv_lines_to_polygons(self, lines) -> List[List[Tuple[float, float]]]:
        """Connect OpenCV detected lines into polygons"""
        polygons = []
        
        try:
            # Convert lines to segments
            segments = []
            scale = 72.0 / self.dpi  # Convert pixels to points
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                start = (x1 * scale, y1 * scale)
                end = (x2 * scale, y2 * scale)
                segments.append((start, end))
            
            # Use same polygon connection logic as DWG parser
            used_segments = set()
            
            for i, segment in enumerate(segments):
                if i in used_segments:
                    continue
                
                polygon = [segment[0], segment[1]]
                current_end = segment[1]
                used_segments.add(i)
                
                max_connections = 15
                connections = 0
                
                while connections < max_connections:
                    found_connection = False
                    
                    for j, other_segment in enumerate(segments):
                        if j in used_segments:
                            continue
                        
                        tolerance = 5.0  # Pixels tolerance
                        
                        if self._points_close_cv(current_end, other_segment[0], tolerance):
                            polygon.append(other_segment[1])
                            current_end = other_segment[1]
                            used_segments.add(j)
                            found_connection = True
                            break
                        elif self._points_close_cv(current_end, other_segment[1], tolerance):
                            polygon.append(other_segment[0])
                            current_end = other_segment[0]
                            used_segments.add(j)
                            found_connection = True
                            break
                    
                    if not found_connection:
                        break
                    
                    if self._points_close_cv(current_end, polygon[0], tolerance):
                        polygon.pop()
                        break
                    
                    connections += 1
                
                if len(polygon) >= 3:
                    polygons.append(polygon)
            
        except Exception as e:
            print(f"Warning: CV line connection failed: {str(e)}")
        
        return polygons

    def _extract_text_based_geometry(self, file_path: str, filename: str) -> List[Dict[str, Any]]:
        """Extract geometry information from text annotations in PDF"""
        zones = []
        
        try:
            doc = fitz.open(file_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get text with coordinates
                text_dict = page.get_text("dict")
                
                # Look for dimension text and coordinate information
                for block in text_dict["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                bbox = span["bbox"]
                                
                                # Look for dimension patterns (e.g., "3.5m", "10'", "250cm")
                                if self._is_dimension_text(text):
                                    # Create a small zone for the dimension
                                    points = [
                                        (bbox[0], bbox[1]),
                                        (bbox[2], bbox[1]),
                                        (bbox[2], bbox[3]),
                                        (bbox[0], bbox[3])
                                    ]
                                    
                                    zone = {
                                        'zone_id': f"text_dim_{page_num}_{len(zones)}",
                                        'zone_type': 'Dimension',
                                        'points': points,
                                        'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                                        'layer': f'Text_Page_{page_num}',
                                        'entity_type': 'TEXT_DIMENSION',
                                        'page': page_num,
                                        'text': text
                                    }
                                    zones.append(zone)
                
                # Look for room labels
                room_labels = self._extract_room_labels(page, page_num)
                zones.extend(room_labels)
            
            doc.close()
            
        except Exception as e:
            print(f"Warning: Text-based extraction failed: {str(e)}")
        
        return zones

    def _extract_room_labels(self, page, page_num: int) -> List[Dict[str, Any]]:
        """Extract room labels and create zones around them"""
        zones = []
        
        try:
            text_dict = page.get_text("dict")
            
            room_keywords = [
                'bedroom', 'kitchen', 'bathroom', 'living', 'dining',
                'chambre', 'cuisine', 'salle', 'salon', 'bureau',
                'office', 'closet', 'hall', 'entry', 'garage'
            ]
            
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip().lower()
                            bbox = span["bbox"]
                            
                            # Check if text contains room keywords
                            for keyword in room_keywords:
                                if keyword in text:
                                    # Create expanded zone around the label
                                    margin = 20  # Expand around text
                                    points = [
                                        (bbox[0] - margin, bbox[1] - margin),
                                        (bbox[2] + margin, bbox[1] - margin),
                                        (bbox[2] + margin, bbox[3] + margin),
                                        (bbox[0] - margin, bbox[3] + margin)
                                    ]
                                    
                                    zone = {
                                        'zone_id': f"room_label_{page_num}_{len(zones)}",
                                        'zone_type': keyword.title(),
                                        'points': points,
                                        'area': (bbox[2] - bbox[0] + 2*margin) * (bbox[3] - bbox[1] + 2*margin),
                                        'layer': f'Room_Labels_Page_{page_num}',
                                        'entity_type': 'ROOM_LABEL',
                                        'page': page_num,
                                        'text': span["text"].strip()
                                    }
                                    zones.append(zone)
                                    break
        
        except Exception as e:
            print(f"Warning: Room label extraction failed: {str(e)}")
        
        return zones

    def _is_dimension_text(self, text: str) -> bool:
        """Check if text represents a dimension"""
        import re
        
        # Patterns for dimensions
        patterns = [
            r'\d+[\.\,]?\d*\s*[mM]',  # meters: 3.5m, 10m
            r'\d+[\.\,]?\d*\s*[cC][mM]',  # centimeters: 250cm
            r'\d+[\.\,]?\d*\s*[mM][mM]',  # millimeters: 2500mm
            r'\d+[\.\,]?\d*\s*[fF][tT]',  # feet: 10ft
            r'\d+[\'\"]',  # feet/inches: 10', 6"
            r'\d+[\.\,]?\d*\s*x\s*\d+[\.\,]?\d*',  # dimensions: 3.5x2.8
        ]
        
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        
        return False

    def _calculate_polygon_area_coords(self, coords: List[Tuple[float, float]]) -> float:
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

    def _points_close_cv(self, p1: Tuple[float, float], p2: Tuple[float, float], tolerance: float = 5.0) -> bool:
        """Check if two points are close enough (for CV processing)"""
        import math
        distance = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        return distance <= tolerance
=======
=======
>>>>>>> origin/replit-agent
import fitz  # PyMuPDF
from typing import List, Dict, Any
from shapely.geometry import Polygon
import logging

class PDFParser:
    """
    Parser for PDF floor plans to extract room geometry and dimensions.
    Uses PyMuPDF to extract vector paths and convert to polygons.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def parse_file(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Parse PDF file and extract zones (closed polygons)

        Args:
            file_bytes: Raw file content as bytes
            filename: Original filename

        Returns:
            List of zone dictionaries with points and metadata
        """
        zones = []
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                shapes = page.get_drawings()
                for shape in shapes:
                    points = []
                    for item in shape["items"]:
                        if item[0] == "l":  # line segment
                            points.append((item[1], item[2]))
                        elif item[0] == "re":  # rectangle
                            x, y, w, h = item[1], item[2], item[3], item[4]
                            points.extend([
                                (x, y),
                                (x + w, y),
                                (x + w, y + h),
                                (x, y + h)
                            ])
                    if points:
                        # Attempt to create polygon and check if closed
                        try:
                            poly = Polygon(points)
                            if poly.is_valid and poly.is_closed and poly.area > 0:
                                zones.append({
                                    "points": points,
                                    "layer": f"Page_{page_num}",
                                    "entity_type": "PDF_SHAPE",
                                    "closed": True,
                                    "area": poly.area
                                })
                        except Exception as e:
                            self.logger.warning(f"Invalid polygon in PDF parsing: {e}")
        except Exception as e:
            self.logger.error(f"Error parsing PDF file {filename}: {e}")
            raise e

        return zones
<<<<<<< HEAD
>>>>>>> origin/replit-agent
=======
>>>>>>> origin/replit-agent
