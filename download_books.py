# download_books.py
import requests
from pathlib import Path
import time
import random
from tqdm import tqdm
import concurrent.futures

class GutenbergDownloader:
    def __init__(self, output_dir='data/raw'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # List of Project Gutenberg mirror URLs
        self.mirrors = [
            'https://www.gutenberg.org/files/',
            'https://mirrors.xmission.com/gutenberg/files/',
            'https://gutenberg.pglaf.org/files/'
        ]
        
        # More extensive book list with genres
        self.books = {
            'classics': [
                1342, 11, 84, 1661, 98, 2701, 1952, 174, 345, 1080,  # Original list
                16, 25344, 1400, 158, 161, 219, 768, 1232, 1260, 2600,
                2701, 3207, 4300, 5200, 120, 1184, 1250, 1268, 2542, 2701,
            ],
            'science_fiction': [
                84, 103, 4368, 8492, 17987, 19827, 21839, 29774, 30827, 35479,
                41445, 42501, 45939, 46587, 51832, 52385, 52504, 53071, 54900, 57861,
            ],
            'mystery': [
                1661, 2852, 3289, 32037, 35598, 36034, 37278, 38205, 39834, 41034,
                42234, 43234, 44234, 45234, 46234, 47234, 48234, 49234, 50234, 51234,
            ],
            'poetry': [
                1065, 1304, 2266, 4300, 4705, 6130, 8387, 15553, 19105, 21839,
                23684, 26715, 30235, 30368, 31647, 34361, 35402, 36021, 37698, 39395,
            ],
            'philosophy': [
                1080, 1497, 2130, 3300, 4705, 5827, 7370, 8438, 10615, 11100,
                13316, 14988, 16712, 18269, 19569, 20833, 22094, 23700, 25717, 28346,
            ]
        }

    def get_mirror_url(self):
        """Get a random mirror URL."""
        return random.choice(self.mirrors)

    def format_book_url(self, book_id):
        """Format the URL for a book based on its ID."""
        return f"{self.get_mirror_url()}{book_id}/{book_id}-0.txt"

    def download_book(self, book_id):
        """Download a single book."""
        url = self.format_book_url(book_id)
        output_path = self.output_dir / f"book_{book_id}.txt"
        
        # Skip if already downloaded
        if output_path.exists():
            return f"Book {book_id} already exists"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            return f"Downloaded book {book_id}"
            
        except Exception as e:
            # Try alternate URL format
            try:
                alt_url = f"{self.get_mirror_url()}{book_id}/{book_id}.txt"
                response = requests.get(alt_url, timeout=30)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                return f"Downloaded book {book_id} (alternate URL)"
                
            except Exception as e2:
                return f"Failed to download book {book_id}: {str(e2)}"

    def download_all_books(self):
        """Download all books using parallel processing."""
        all_books = []
        for genre, books in self.books.items():
            all_books.extend(books)
        
        # Remove duplicates
        all_books = list(set(all_books))
        print(f"Downloading {len(all_books)} books...")
        
        # Use ThreadPoolExecutor for parallel downloads
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self.download_book, book_id) for book_id in all_books]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_books)):
                print(future.result())
                time.sleep(0.5)  # Be nice to the servers

def main():
    downloader = GutenbergDownloader()
    downloader.download_all_books()
    print("\nDownload complete! Now you can run data_processor.py")

if __name__ == "__main__":
    main()
