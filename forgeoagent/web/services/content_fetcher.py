#!/usr/bin/env python3
"""
Content Image Fetcher Service

This module provides functionality to fetch content with images using multiple sources:
- requests-html (lightweight JS rendering) for high-quality images with automatic fallback
- Page source parsing for fast, lightweight extraction
- Gemini API for AI-powered image suggestions (optional)

File: services/content_fetcher.py
"""

import base64
import requests
from typing import List, Dict, Optional
from forgeoagent.clients.gemini_engine import GeminiAPIClient
from google.genai import types
from google import genai
import json
from urllib.parse import quote
import logging
from bs4 import BeautifulSoup
import time
from requests_html import HTMLSession

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentImageFetcher:
    """
    Unified image fetcher that combines multiple extraction methods:
    - requests-html session (lightweight JS rendering, with automatic fallback)
    - Page source parsing (fast, lightweight)
    - Gemini API (AI-powered suggestions, optional)
    
    Automatically tries session extraction first, falls back to page source if it fails.
    Supports downloading images and converting them to base64 format.
    """
    
    def __init__(self, gemini_api_keys: List[str], headless: bool = True, default_timeout_ms: int = 30000):
        """
        Initialize the ContentImageFetcher with Gemini API credentials and session settings.
        
        Args:
            gemini_api_keys: List of Gemini API keys for authentication and load balancing
            headless: Whether to run browser in headless mode (no visible window)
            default_timeout_ms: Default timeout for browser operations in milliseconds
        """
        self.gemini_api_keys = gemini_api_keys
        self.headless = headless
        self.default_timeout_seconds = default_timeout_ms / 1000  # Convert to seconds
        self.session: Optional[HTMLSession] = None
        
        self.gemini_system_prompt = (
            "Give relevant topic working images links and title based on the given description. "
            "Output format : {'images_links':[link1,link2],'main_title':relevant_title,'response':relevant_response}"
        )
        
        # Define output schema for structured Gemini responses
        self.gemini_output_schema_properties = {
            "response": types.Schema(
                type=genai.types.Type.STRING, 
                description="The agent's response to the given task"
            ),
            "main_title": types.Schema(
                type=genai.types.Type.STRING, 
                description="The main title of the content"
            ),
            "images_links": types.Schema(
                type=genai.types.Type.ARRAY,
                items=types.Schema(type=genai.types.Type.STRING),
                description="List of image links related to the topic"
            )
        }
        self.gemini_output_required_fields = ["response", "main_title", "images_links"]
        
        # Initialize Gemini client
        self.gemini_client = GeminiAPIClient(
            system_instruction=self.gemini_system_prompt,
            api_keys=self.gemini_api_keys,
            # output_properties=self.gemini_output_schema_properties,
            # output_required=self.gemini_output_required_fields
        )
    
    def _initialize_session(self):
        """
        Initialize the requests-html session.
        
        Creates a new HTMLSession if not already initialized.
        """
        if not self.session:
            self.session = HTMLSession()
            logger.info("HTML session started successfully")
    
    def _terminate_session(self):
        """
        Close and cleanup the session instance.
        
        Properly closes the session and resets instance variables.
        """
        if self.session:
            self.session.close()
            self.session = None
            logger.info("Session closed")
    
    def _extract_high_quality_images(
        self, 
        search_query: str, 
        max_image_count: int = 10
    ) -> List[Dict[str, str]]:
        """
        Extract high-quality images from Google Images using requests-html.
        
        This method uses requests-html to:
        1. Navigate to Google Images with the search query
        2. Render JavaScript to load dynamic content
        3. Extract image URLs and metadata from the rendered page
        
        Args:
            search_query: The search query for Google Images
            max_image_count: Maximum number of images to extract
            
        Returns:
            List of dictionaries containing:
                - image_url: High-quality image URL
                - image_title: Title/alt text of the image
                - source_url: Source webpage URL where the image originates
        """
        extracted_images_data = []
        
        try:
            self._initialize_session()
            
            # Navigate to Google Images
            google_images_url = f"https://www.google.com/search?q={quote(search_query)}&tbm=isch"
            logger.info(f"Fetching: {google_images_url}")
            
            response = self.session.get(google_images_url)
            
            # Render JavaScript to load dynamic content
            logger.info("Rendering JavaScript content...")
            response.html.render(timeout=int(self.default_timeout_seconds), sleep=2)
            
            # Find all image elements
            img_elements = response.html.find('img')
            logger.info(f"Found {len(img_elements)} image elements")
            
            # Extract image data
            for img in img_elements:
                if len(extracted_images_data) >= max_image_count:
                    break
                
                try:
                    # Get image attributes
                    img_src = img.attrs.get('src', '') or img.attrs.get('data-src', '')
                    img_alt = img.attrs.get('alt', 'No title')
                    
                    # Skip if no src or if it's a data URI or too short alt text
                    if not img_src or img_src.startswith('data:') or len(img_alt) < 3:
                        continue
                    
                    # Skip Google UI elements (logo, icons, etc.)
                    if 'logo' in img_alt.lower() or 'icon' in img_alt.lower():
                        continue
                    
                    # Try to find parent link for source URL
                    source_url = ''
                    parent = img.element.getparent()
                    while parent is not None:
                        if parent.tag == 'a' and 'href' in parent.attrib:
                            source_url = parent.attrib['href']
                            break
                        parent = parent.getparent()
                    
                    logger.info(f"Extracted image: {img_src[:80]}...")
                    extracted_images_data.append({
                        'image_url': img_src,
                        'image_title': img_alt,
                        'source_url': source_url,
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing image element: {e}")
                    continue
            
            logger.info(f"Successfully extracted {len(extracted_images_data)} images via requests-html")
            
        except Exception as e:
            logger.error(f"Error during image extraction: {e}")
            raise  # Re-raise to trigger fallback
        finally:
            self._terminate_session()
        
        return extracted_images_data
    
    def _extract_images_with_session_sync(
        self, 
        search_query: str, 
        max_image_count: int = 10
    ) -> List[Dict[str, str]]:
        """
        Synchronous session-based image extraction using requests-html.
        
        Args:
            search_query: The search query for Google Images
            max_image_count: Maximum number of images to extract
            
        Returns:
            List of dictionaries with image data (image_url, image_title, source_url)
        """
        return self._extract_high_quality_images(search_query, max_image_count)
    
    def download_image_as_base64(self, image_url: str, request_timeout_seconds: int = 10) -> Optional[str]:
        """
        Download an image from a URL and convert it to base64 data URI format.
        
        Args:
            image_url: The URL of the image to download
            request_timeout_seconds: Maximum time to wait for the download request in seconds
            
        Returns:
            Base64 encoded data URI string (e.g., 'data:image/jpeg;base64,...'), or None if download failed
        """
        try:
            response = requests.get(image_url, timeout=request_timeout_seconds, stream=True)
            response.raise_for_status()
            
            # Read image content
            image_binary_content = response.content
            
            # Convert to base64
            base64_encoded_string = base64.b64encode(image_binary_content).decode('utf-8')
            
            # Get content type for data URI
            content_type = response.headers.get('Content-Type', 'image/jpeg')
            
            # Return as data URI format
            return f"data:{content_type};base64,{base64_encoded_string}"
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching image from {image_url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return None
  
    def extract_images_from_google_page_source(
        self, 
        search_query: str, 
        max_image_count: int = 10, 
        page_start_percentage: float = 30.0
    ) -> List[Dict]:
        """
        Fetch Google Images page source and extract image data with metadata.
        
        This method parses the HTML page source to extract images without browser automation.
        It's faster but may return lower quality images compared to browser-based extraction.
        
        Args:
            search_query: The search query for Google Images
            max_image_count: Maximum number of images to extract
            page_start_percentage: Percentage of page to skip before parsing (to avoid UI elements)
        
        Returns:
            List of dictionaries containing:
                - image_data: base64 encoded image or image URL
                - image_title: title/alt text of the image from Google search
                - source_url: webpage link where the image originates from
                - is_base64: boolean indicating if image_data is base64 encoded or a URL
        """
        try:
            search_url = f"https://www.google.com/search?q={quote(search_query)}&tbm=isch"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            logger.info(f"Fetching page source from: {search_url}")
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            page_html_content = response.text
            logger.info(f"Page source fetched. Total length: {len(page_html_content)} characters")
            
            # Calculate start position to skip header/UI elements
            start_position = int(len(page_html_content) * (page_start_percentage / 100.0))
            page_section_to_parse = page_html_content[start_position:]
            
            soup = BeautifulSoup(page_section_to_parse, 'html.parser')
            
            extracted_images_data = []
            seen_image_urls = set()
            
            # Find all anchor tags containing images
            for anchor_tag in soup.find_all('a', href=True):
                href_value = anchor_tag.get('href', '')
                
                # Skip internal search links
                if href_value.startswith('/search?'):
                    continue
                
                # Find img tags within this anchor
                image_tags = anchor_tag.find_all('img', src=True)
                
                for image_tag in image_tags:
                    image_src = image_tag.get('src', '')
                    image_alt_text = image_tag.get('alt', 'No title available')
                    
                    # Create unique key to avoid duplicates
                    unique_identifier = f"{href_value}_{image_src[:50]}"
                    
                    if unique_identifier not in seen_image_urls:
                        seen_image_urls.add(unique_identifier)
                        
                        image_metadata = {
                            'image_title': image_alt_text,
                            'source_url': href_value,
                            'is_base64': False,
                            'image_data': None
                        }
                        
                        # Check if image src is already base64 encoded
                        if image_src.startswith('data:'):
                            image_metadata['image_data'] = image_src
                            image_metadata['is_base64'] = True
                            logger.info(f"Found base64 image with title: {image_alt_text[:50]}...")
                        else:
                            # It's a URL, try to convert to base64
                            base64_image_data = self.download_image_as_base64(image_src)
                            if base64_image_data:
                                image_metadata['image_data'] = base64_image_data
                                image_metadata['is_base64'] = True
                                logger.info(f"Converted URL to base64 for: {image_alt_text[:50]}...")
                            else:
                                # Keep as URL if conversion fails
                                image_metadata['image_data'] = image_src
                                image_metadata['is_base64'] = False
                                logger.warning(f"Failed to convert, keeping URL for: {image_alt_text[:50]}...")
                        
                        extracted_images_data.append(image_metadata)
                        
                        # Stop if we've reached max_image_count
                        if len(extracted_images_data) >= max_image_count:
                            break
                
                if len(extracted_images_data) >= max_image_count:
                    break
            
            logger.info(f"Extracted {len(extracted_images_data)} images from page source")
            return extracted_images_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching page source: {e}")
            return []
        except Exception as e:
            logger.error(f"Error extracting images from page source: {e}")
            return []
    
    def _fetch_images_from_gemini_api(
        self,
        search_query: str,
        convert_to_base64: bool = True
    ) -> Dict:
        """
        Fetch image suggestions from Gemini API.
        
        Args:
            search_query: The search query for image suggestions
            convert_to_base64: Whether to download and convert images to base64
            
        Returns:
            Dictionary containing:
                - response: Gemini's text response
                - main_title: Suggested main title
                - images_links: List of image URLs
                - images_base64: List of base64 encoded images (if convert_to_base64=True)
                - failed_images: List of URLs that failed to convert
                - error: Error message if request failed
        """
        try:
            gemini_response_text = self.gemini_client.search_content(
                prompt=search_query,
                system_instruction=self.gemini_system_prompt
            )
            gemini_response_json = json.loads(gemini_response_text.replace("```json", "").replace("```", ""))
            logger.info(f"Gemini response: {gemini_response_json}")
            
            # Initialize Gemini result
            gemini_result = {
                "response": gemini_response_json.get("response", ""),
                "main_title": gemini_response_json.get("main_title", ""),
                "images_links": gemini_response_json.get("images_links", []),
            }
            
            # Convert Gemini images to base64 if requested
            if convert_to_base64 and gemini_result.get("images_links", []):
                gemini_result["images_base64"] = []
                gemini_result["failed_images"] = []
                
                for image_url in gemini_result["images_links"]:
                    base64_image_data = self.download_image_as_base64(image_url)
                    if base64_image_data:
                        gemini_result["images_base64"].append(base64_image_data)
                    else:
                        gemini_result["failed_images"].append(image_url)

            return gemini_result
            
        except Exception as e:
            logger.error(f"Error fetching images from Gemini: {e}")
            return {"error": str(e)}

    def fetch_images_for_content(
        self, 
        content_title: str, 
        content_description: Optional[str] = None,
        convert_to_base64: bool = True,
        use_gemini_api: bool = False,
        max_images_per_source: int = 5,
        run_browser_headless: bool = True
    ) -> Dict:
        """
        Fetch images for specific content using multiple sources with automatic fallback.
        
        Extraction strategy:
        1. Try requests-html extraction (lightweight with JS rendering) first
        2. If that fails, automatically fallback to page source parsing (fast)
        3. Optionally fetch AI-powered suggestions from Gemini API
        
        Args:
            content_title: Main title for the content
            content_description: Additional description to refine image search (optional)
            convert_to_base64: Whether to convert images to base64 data URIs
            use_gemini_api: Whether to fetch image suggestions from Gemini API
            max_images_per_source: Maximum number of images to extract from each source
            run_browser_headless: Whether to run in headless mode (for session extraction)
        
        Returns:
            Dictionary containing:
                - images_data: List of images (from session or page source fallback)
                - extraction_method: 'session' or 'page_source' indicating which method succeeded
                - gemini_response: Gemini API response with image suggestions (if enabled)
                - all_images_data: Combined list of all base64 encoded images
                - all_images_links: Combined list of all image URLs
        """
        search_query = f"Title: {content_title}"
        if content_description:
            search_query += f"\nDescription: {content_description}"
        
        result = {}
        images_data = []
        extraction_method = None
        
        # Update headless setting
        self.headless = run_browser_headless
        
        # Try session extraction first, fallback to page source on failure
        try:
            logger.info("Attempting requests-html session-based image extraction...")
            session_images = self._extract_images_with_session_sync(
                search_query=search_query,
                max_image_count=max_images_per_source
            )
            
            if session_images:
                # Convert to base64 if requested
                for img_info in session_images:
                    image_url = img_info.get('image_url', '')
                    image_title = img_info.get('image_title', 'No title')
                    source_url = img_info.get('source_url', '')
                    
                    image_metadata = {
                        'image_title': image_title,
                        'source_url': source_url,
                        'is_base64': False,
                        'image_data': image_url
                    }
                    
                    if convert_to_base64 and image_url:
                        base64_image_data = self.download_image_as_base64(image_url)
                        if base64_image_data:
                            image_metadata['image_data'] = base64_image_data
                            image_metadata['is_base64'] = True
                            logger.info(f"Converted session image to base64: {image_title[:50]}...")
                        else:
                            logger.warning(f"Failed to convert to base64, keeping URL: {image_title[:50]}...")
                    
                    images_data.append(image_metadata)
                
                extraction_method = 'session'
                logger.info(f"Session extraction successful: {len(images_data)} images")
            else:
                raise Exception("Session extraction returned no images")
                
        except Exception as e:
            logger.warning(f"Session extraction failed: {e}. Falling back to page source extraction...")
            
            # Fallback to page source extraction
            search_query_with_hint = search_query + "\n Give relevant images valid links for this topic from google search "
            images_data = self.extract_images_from_google_page_source(
                search_query_with_hint, 
                max_image_count=max_images_per_source,
                page_start_percentage=30.0
            )
            extraction_method = 'page_source'
            logger.info(f"Page source extraction completed: {len(images_data)} images")
        
        result["images_data"] = images_data
        result["extraction_method"] = extraction_method
        
        # Gemini API extraction (optional)
        if use_gemini_api:
            search_query_with_hint = search_query + "\n Give relevant images valid links for this topic from google search "
            gemini_result = self._fetch_images_from_gemini_api(
                search_query_with_hint,
                convert_to_base64=convert_to_base64
            )
            result["gemini_response"] = gemini_result
        
        # Aggregate all images
        result["all_images_data"] = []
        result["all_images_links"] = []

        # Add Gemini images to aggregated lists
        if use_gemini_api and "gemini_response" in result and result["gemini_response"]:
            gemini_response = result["gemini_response"]
            if "images_base64" in gemini_response:
                result["all_images_data"].extend(gemini_response["images_base64"])
            if "images_links" in gemini_response:
                result["all_images_links"].extend(gemini_response["images_links"])

        # Add extracted images to aggregated lists
        if images_data:
            for image_info in images_data:
                if image_info.get('is_base64'):
                    result["all_images_data"].append(image_info['image_data'])
                else:
                    result["all_images_links"].append(image_info['image_data'])
        
        return result

# Standalone function for quick usage
def fetch_content_images(
    content_title: str,
    content_description: Optional[str] = None,
    gemini_api_keys: Optional[List[str]] = None,
    convert_to_base64: bool = True,
    use_gemini_api: bool = False,
    max_images_per_source: int = 10,
    run_browser_headless: bool = True
) -> Dict:
    """
    Standalone function to fetch and convert images for given content.
    
    Automatically tries requests-html session extraction first, falls back to page source if it fails.
    
    Args:
        content_title: Main title for the content
        content_description: Additional description to refine image search (optional)
        gemini_api_keys: List of Gemini API keys for authentication
        convert_to_base64: Whether to convert images to base64 data URIs
        use_gemini_api: Whether to fetch image suggestions from Gemini API
        max_images_per_source: Maximum images to extract from each source (default: 10)
        run_browser_headless: Whether to run in headless mode
        
    Returns:
        Dictionary containing images from various sources and aggregated lists
        
    Raises:
        ValueError: If gemini_api_keys is not provided
    """
    if not gemini_api_keys:
        raise ValueError("API keys are required")
    
    fetcher = ContentImageFetcher(gemini_api_keys=gemini_api_keys, headless=run_browser_headless)
    result = fetcher.fetch_images_for_content(
        content_title=content_title, 
        content_description=content_description,
        convert_to_base64=convert_to_base64,
        use_gemini_api=use_gemini_api,
        max_images_per_source=max_images_per_source,
        run_browser_headless=run_browser_headless
    )
    
    return result

if __name__ == "__main__":
    # Example usage with automatic fallback
    API_KEYS = ["xx"]
    title = "nature photography"
    description = "Stunning landscapes and wildlife photography"
    
    print("=" * 80)
    print("AUTOMATIC EXTRACTION (requests-html with Page Source Fallback)")
    print("=" * 80)
    result = fetch_content_images(
        content_title=title,
        content_description=description,
        gemini_api_keys=API_KEYS,
        convert_to_base64=True,
        max_images_per_source=3,
        run_browser_headless=False
    )
    
    print(f"\nExtraction method used: {result.get('extraction_method')}")
    print(f"Extracted {len(result.get('images_data', []))} images")
    print(json.dumps(result, indent=2))