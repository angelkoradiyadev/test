#!/usr/bin/env python3
"""
Content Image Fetcher Service

This module provides functionality to fetch content with images using multiple sources:
- Requests/BeautifulSoup for high-quality images with automatic fallback
- Page source parsing for fast, lightweight extraction
- Gemini API for AI-powered image suggestions (optional)

File: services/content_fetcher.py
"""

import base64
import requests
import re
from typing import List, Dict, Optional
from forgeoagent.clients.gemini_engine import GeminiAPIClient
from google.genai import types
from google import genai
import json
from urllib.parse import quote, urlparse, parse_qs
import logging
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentImageFetcher:
    """
    Unified image fetcher that combines multiple extraction methods:
    - Requests/BeautifulSoup extraction (fast, no browser needed)
    - Page source parsing (lightweight fallback)
    - Gemini API (AI-powered suggestions, optional)
    
    Automatically tries requests extraction first, falls back to page source if it fails.
    Supports downloading images and converting them to base64 format.
    """
    
    def __init__(self, gemini_api_keys: List[str]):
        """
        Initialize the ContentImageFetcher with Gemini API credentials.
        
        Args:
            gemini_api_keys: List of Gemini API keys for authentication and load balancing
        """
        self.gemini_api_keys = gemini_api_keys
        
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
    

    
    def _extract_high_quality_images_sync(
        self, 
        search_query: str, 
        max_image_count: int = 10
    ) -> List[Dict[str, str]]:
        """
        Extract high-quality images from Google Images using requests and BeautifulSoup.
        
        This method:
        1. Navigates to Google Images with the search query
        2. Parses embedded JavaScript data containing image URLs
        3. Extracts high-resolution image URLs from the data
        4. Collects metadata (title, source URL)
        
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
            # Navigate to Google Images
            google_images_url = f"https://www.google.com/search?q={quote(search_query)}&tbm=isch"
            logger.info(f"Fetching images from: {google_images_url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://www.google.com/'
            }
            
            response = requests.get(google_images_url, headers=headers, timeout=10)
            response.raise_for_status()
            print(response.text)
            page_content = response.text
            
            # Extract image URLs from embedded JavaScript data
            # Google embeds image data in JavaScript objects in the page
            
            # Pattern to find image URLs in the JavaScript data
            # Look for patterns like ["https://...jpg",width,height]
            url_pattern = r'https?://[^\s\'"<>]+?\.(?:jpg|jpeg|png|gif|webp)'
            
            # Find all potential image URLs
            found_urls = re.findall(url_pattern, page_content, re.IGNORECASE)
            
            logger.info(f"Found {len(found_urls)} potential image URLs in page source")
            
            # Filter and deduplicate URLs
            seen_urls = set()
            successfully_extracted_count = 0
            
            for url in found_urls:
                if successfully_extracted_count >= max_image_count:
                    break
                
                # Clean up the URL (remove any trailing characters)
                url = url.rstrip('\\",;)}]')
                
                # Skip if we've seen this URL or if it's too short
                if url in seen_urls or len(url) < 20:
                    continue
                
                # Skip Google's own UI images and thumbnails
                if any(skip in url.lower() for skip in ['gstatic.com/images', 'google.com/images', 'encrypted-tbn']):
                    continue
                
                seen_urls.add(url)
                
                logger.info(f"Extracted image {successfully_extracted_count + 1}: {url[:80]}...")
                
                extracted_images_data.append({
                    'image_url': url,
                    'image_title': 'Image from Google Search',
                    'source_url': '',
                })
                successfully_extracted_count += 1
            
            # If we didn't get enough images, try parsing img tags as fallback
            if successfully_extracted_count < max_image_count:
                logger.info("Attempting to extract images from img tags as fallback...")
                soup = BeautifulSoup(page_content, 'html.parser')
                img_tags = soup.find_all('img')
                
                for img_tag in img_tags:
                    if successfully_extracted_count >= max_image_count:
                        break
                    
                    img_src = img_tag.get('src', '') or img_tag.get('data-src', '')
                    img_alt = img_tag.get('alt', 'No title')
                    
                    # Skip if already processed or invalid
                    if not img_src or img_src in seen_urls:
                        continue
                    
                    # Skip data URIs, small images, and Google UI elements
                    if (img_src.startswith('data:') or 
                        'gstatic.com/images' in img_src or 
                        'encrypted-tbn' in img_src or
                        len(img_alt) < 3):
                        continue
                    
                    if img_src.startswith('http'):
                        seen_urls.add(img_src)
                        
                        # Try to find parent link for source URL
                        parent_link = img_tag.find_parent('a')
                        source_url = parent_link.get('href', '') if parent_link else ''
                        
                        extracted_images_data.append({
                            'image_url': img_src,
                            'image_title': img_alt,
                            'source_url': source_url,
                        })
                        successfully_extracted_count += 1
                        logger.info(f"Extracted image from img tag: {img_alt[:50]}...")
            
            logger.info(f"Successfully extracted {len(extracted_images_data)} images using requests/BeautifulSoup")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error during HTTP request: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during image extraction: {e}")
            raise
        
        return extracted_images_data
    
    def _extract_images_with_browser_sync(
        self, 
        search_query: str, 
        max_image_count: int = 10
    ) -> List[Dict[str, str]]:
        """
        Synchronous wrapper for image extraction using requests and BeautifulSoup.
        
        Args:
            search_query: The search query for Google Images
            max_image_count: Maximum number of images to extract
            
        Returns:
            List of dictionaries with image data (image_url, image_title, source_url)
        """
        return self._extract_high_quality_images_sync(search_query, max_image_count)
    
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
        return [] # TEMP
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
        max_images_per_source: int = 5
    ) -> Dict:
        """
        Fetch images for specific content using multiple sources with automatic fallback.
        
        Extraction strategy:
        1. Try requests/BeautifulSoup extraction (fast, no browser needed)
        2. If that fails, fallback to page source parsing
        3. Optionally fetch AI-powered suggestions from Gemini API
        
        Args:
            content_title: Main title for the content
            content_description: Additional description to refine image search (optional)
            convert_to_base64: Whether to convert images to base64 data URIs
            use_gemini_api: Whether to fetch image suggestions from Gemini API
            max_images_per_source: Maximum number of images to extract from each source
        
        Returns:
            Dictionary containing:
                - images_data: List of images from extraction
                - extraction_method: 'requests' or 'page_source' indicating which method succeeded
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
        
        # Try requests/BeautifulSoup extraction first, fallback to page source on failure
        try:
            logger.info("Attempting requests/BeautifulSoup image extraction...")
            extracted_images = self._extract_images_with_browser_sync(
                search_query=search_query,
                max_image_count=max_images_per_source
            )
            
            if extracted_images:
                # Convert to base64 if requested
                for img_info in extracted_images:
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
                            logger.info(f"Converted image to base64: {image_title[:50]}...")
                        else:
                            logger.warning(f"Failed to convert to base64, keeping URL: {image_title[:50]}...")
                    
                    images_data.append(image_metadata)
                
                extraction_method = 'requests'
                logger.info(f"Requests extraction successful: {len(images_data)} images")
            else:
                raise Exception("Requests extraction returned no images")
                
        except Exception as e:
            logger.warning(f"Requests extraction failed: {e}. Falling back to page source extraction...")
            
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
    max_images_per_source: int = 10
) -> Dict:
    """
    Standalone function to fetch and convert images for given content.
    
    Uses requests/BeautifulSoup for image extraction (no browser needed).
    
    Args:
        content_title: Main title for the content
        content_description: Additional description to refine image search (optional)
        gemini_api_keys: List of Gemini API keys for authentication
        convert_to_base64: Whether to convert images to base64 data URIs
        use_gemini_api: Whether to fetch image suggestions from Gemini API
        max_images_per_source: Maximum images to extract from each source (default: 10)
        
    Returns:
        Dictionary containing images from various sources and aggregated lists
        
    Raises:
        ValueError: If gemini_api_keys is not provided
    """
    if not gemini_api_keys:
        raise ValueError("API keys are required")
    
    fetcher = ContentImageFetcher(gemini_api_keys=gemini_api_keys)
    result = fetcher.fetch_images_for_content(
        content_title=content_title, 
        content_description=content_description,
        convert_to_base64=convert_to_base64,
        use_gemini_api=use_gemini_api,
        max_images_per_source=max_images_per_source
    )
    
    return result

if __name__ == "__main__":
    # Example usage with requests/BeautifulSoup extraction
    API_KEYS = ["xx"]
    title = "nature photography"
    description = "Stunning landscapes and wildlife photography"
    
    print("=" * 80)
    print("AUTOMATIC EXTRACTION (Requests/BeautifulSoup with Page Source Fallback)")
    print("=" * 80)
    result = fetch_content_images(
        content_title=title,
        content_description=description,
        gemini_api_keys=API_KEYS,
        convert_to_base64=True,
        max_images_per_source=3
    )
    
    print(f"\nExtraction method used: {result.get('extraction_method')}")
    print(f"Extracted {len(result.get('images_data', []))} images")
    print(json.dumps(result, indent=2))