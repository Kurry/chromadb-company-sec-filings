#!/usr/bin/env python3
"""
SEC Filings ChromaDB Builder

This script downloads SEC filings for a specified company ticker, extracts structured 
data using edgartools' Data Objects, and stores them in a ChromaDB for semantic search 
and retrieval.
"""

import os
import time
import logging
import argparse
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import re
from functools import lru_cache
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import required packages
from edgar import Company, set_identity
from edgar.xbrl.xbrl import XBRL
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Compile regex patterns at module scope for better performance
PRESS_RELEASE_PATTERN = re.compile(r".*RELEASE", re.IGNORECASE)
EXHIBIT_PATTERN = re.compile(r"EX-", re.IGNORECASE)

class FilingMetadata(BaseModel):
    """Metadata for a filing section."""
    ticker: str
    cik: str
    company_name: str
    filing_type: str
    filing_date: str
    filing_period: str
    filing_url: str
    section_id: str
    section_title: str
    section_level: int
    parent_section_title: Optional[str] = None
    statement_type: Optional[str] = None
    item_number: Optional[str] = None
    # Additional KPI fields for financial data
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    profit_margin: Optional[float] = None
    html_anchor: Optional[str] = None
    # Time series data (as JSON strings)
    revenue_ts: Optional[str] = None
    net_income_ts: Optional[str] = None
    eps_ts: Optional[str] = None
    # XBRL context information
    context_ref: Optional[str] = None
    unit: Optional[str] = None
    statement_row: Optional[str] = None  # For row-level chunking
    xbrl_fact_id: Optional[str] = None  # Original fact ID for traceability
    part: Optional[str] = None
    item: Optional[str] = None
    content_type: Optional[str] = None
    chunk_type: Optional[str] = None


class SECFilingProcessor:
    """Process SEC filings into a ChromaDB using Data Objects from edgartools."""
    
    def __init__(
        self,
        ticker: str,
        filing_types: List[str] = ["10-K", "10-Q", "8-K"],
        years: int = 3,
        output_dir: str = "./sec_db",
        embedding_model: str = "FinLang/investopedia_embedding",
        batch_size: int = 32,
        overwrite: bool = False,
        include_exhibits: bool = False
    ):
        """Initialize the SEC Filing Processor."""
        self.ticker = ticker.upper()
        self.filing_types = filing_types
        self.years = years
        self.output_dir = Path(output_dir)
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.overwrite = overwrite
        self.include_exhibits = include_exhibits
        
        # Create the output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking variables
        self.company = None
        self.cik = None
        self.company_name = None
        self.processed_filings = set()
        
        # Initialize ChromaDB client and collection
        self.initialize_chroma()
        
        # Initialize text splitter for chunking
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def initialize_chroma(self) -> None:
        """Initialize ChromaDB client and collection."""
        logger.info(f"Initializing ChromaDB in: {self.output_dir}")
        
        # Initialize embedding function
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.output_dir))
        
        # Get or create collection
        collection_name = f"{self.ticker}_filings"
        
        collections = self.client.list_collections()
        collection_names = [c.name for c in collections]
        
        if collection_name in collection_names:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Using existing collection '{collection_name}' with {self.collection.count()} documents")
            
            # Get existing document IDs
            if self.collection.count() > 0:
                existing_docs = self.collection.get()
                for metadata in existing_docs["metadatas"]:
                    if metadata and "filing_url" in metadata:
                        self.processed_filings.add(metadata["filing_url"])
                
                logger.info(f"Found {len(self.processed_filings)} previously processed filings")
        else:
            # Create new collection
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Created new collection '{collection_name}'")
    
    def get_company_info(self) -> None:
        """Get company information and CIK."""
        logger.info(f"Fetching company information for {self.ticker}")
        
        # Use Company object directly from edgartools
        self.company = Company(self.ticker)
        self.company_name = self.company.name
        self.cik = self.company.cik
        
        logger.info(f"Company: {self.company_name} (CIK: {self.cik})")
    
    def get_filings(self) -> List[Any]:
        """Get filings for the company using edgartools with pre-loaded objects."""
        if not self.company:
            self.get_company_info()
        
        logger.info(f"Fetching {', '.join(self.filing_types)} filings for {self.company_name} (past {self.years} years)")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * self.years)
        date_range = f"{start_date.strftime('%Y-%m-%d')}:{end_date.strftime('%Y-%m-%d')}"
        
        # Get all filings in a single call
        filings = self.company.get_filings(form=self.filing_types)
        filings = filings.filter(filing_date=date_range)
        
        logger.info(f"Found {len(filings)} total filings")
        return filings
    
    def process_filing(self, filing) -> List[Tuple[str, Dict[str, Any]]]:
        """Process a single filing using edgartools Data Objects with unified sections interface."""
        # Skip if already processed
        filing_url = filing.url
        if filing_url in self.processed_filings and not self.overwrite:
            logger.info(f"Skipping already processed filing: {filing_url}")
            return []
        
        filing_type = filing.form
        filing_date = filing.filing_date
        filing_period = getattr(filing, "period_of_report", filing_date)
        
        logger.info(f"Processing {filing_type} filing from {filing_date}")
        
        processed_chunks = []
        
        try:
            # Create data object from filing using the obj() method (now cached from with_objects())
            data_object = filing.obj()
            
            if data_object:
                logger.info(f"Created Data Object of type: {type(data_object).__name__}")
                
                # 1. Unified sections handling for all report types
                if hasattr(data_object, "sections"):
                    for sec_id, sec in data_object.sections.items():
                        if not sec.text or len(sec.text.strip()) < 10:
                            continue
                        
                        # Create metadata for section with more hierarchy information
                        section_metadata = FilingMetadata(
                            ticker=self.ticker,
                            cik=str(self.cik),
                            company_name=self.company_name,
                            filing_type=filing_type,
                            filing_date=str(filing_date),
                            filing_period=str(filing_period),
                            filing_url=filing_url,
                            section_id=sec_id,
                            section_title=sec.title,
                            section_level=sec.level,
                            parent_section_title=getattr(sec, "parent_title", None),
                            statement_type=None,
                            item_number=getattr(sec, "item_number", None),
                            html_anchor=getattr(sec, "html_anchor", ""),
                            part=getattr(sec, "part_title", None),  # Add part title
                            item=getattr(sec, "item_title", None)   # Add item title
                        ).dict()
                        
                        # Process both chunks and sentences for better embedding
                        # 1. Regular chunks for context
                        chunks = self.splitter.split_text(sec.text)
                        for i, chunk in enumerate(chunks):
                            chunk_metadata = section_metadata.copy()
                            chunk_metadata["chunk_index"] = i
                            chunk_metadata["chunk_count"] = len(chunks)
                            chunk_metadata["chunk_type"] = "paragraph"
                            processed_chunks.append((chunk, chunk_metadata))
                        
                        # 2. Sentences for targeted Q&A matches (if available)
                        if hasattr(data_object, "doc") and hasattr(data_object.doc, "sentences"):
                            try:
                                sentences = data_object.doc.sentences(sec_id=sec_id, max_len=80)
                                for i, sentence in enumerate(sentences):
                                    if len(sentence.strip()) > 15:  # Skip short sentences
                                        sent_metadata = section_metadata.copy()
                                        sent_metadata["chunk_index"] = i
                                        sent_metadata["chunk_count"] = len(sentences)
                                        sent_metadata["chunk_type"] = "sentence"
                                        processed_chunks.append((sentence, sent_metadata))
                            except Exception as e:
                                logger.warning(f"Error extracting sentences: {str(e)}")
                
                # 2. Use the summary helper for narrative MD&A if available
                if hasattr(data_object, "summary"):
                    try:
                        mda_summary = data_object.summary(kind="md&a")
                        if mda_summary and len(mda_summary.strip()) > 10:
                            summary_metadata = FilingMetadata(
                                ticker=self.ticker,
                                cik=str(self.cik),
                                company_name=self.company_name,
                                filing_type=filing_type,
                                filing_date=str(filing_date),
                                filing_period=str(filing_period),
                                filing_url=filing_url,
                                section_id="mda-summary",
                                section_title="Management Discussion & Analysis Summary",
                                section_level=1,
                                parent_section_title=None,
                                statement_type=None,
                                item_number=None,
                                chunk_type="summary"
                            ).dict()
                            processed_chunks.append((mda_summary, summary_metadata))
                    except Exception as e:
                        logger.warning(f"Error getting MD&A summary: {str(e)}")
                
                # 3. Use press_releases property directly for 8-K
                if filing_type == "8-K" and hasattr(data_object, "press_releases"):
                    releases = data_object.press_releases
                    if releases:
                        for i, press_release in enumerate(releases):
                            pr_metadata = FilingMetadata(
                                ticker=self.ticker,
                                cik=str(self.cik),
                                company_name=self.company_name,
                                filing_type=filing_type,
                                filing_date=str(filing_date),
                                filing_period=str(filing_period),
                                filing_url=filing_url,
                                section_id=f"press-release-{i+1}",
                                section_title=f"Press Release {i+1}",
                                section_level=1,
                                parent_section_title=None,
                                statement_type=None,
                                item_number=None,
                                chunk_type="press_release"
                            ).dict()
                            
                            # Split content into chunks
                            chunks = self.splitter.split_text(str(press_release))
                            
                            for j, chunk in enumerate(chunks):
                                chunk_metadata = pr_metadata.copy()
                                chunk_metadata["chunk_index"] = j
                                chunk_metadata["chunk_count"] = len(chunks)
                                processed_chunks.append((chunk, chunk_metadata))
                
                # 4. 13F Holdings Report - use DataFrame conversion
                elif filing_type == "13F-HR" and hasattr(data_object, "infotable"):
                    # Get holdings data as a DataFrame and convert to markdown
                    try:
                        holdings_df = data_object.infotable.to_dataframe()
                        holdings_text = holdings_df.to_markdown()
                    except:
                        holdings_text = str(data_object.infotable)
                    
                    # Add total value and count metrics
                    holdings_summary = f"Total Holdings: {getattr(data_object, 'total_holdings', 'N/A')}\n"
                    holdings_summary += f"Total Value: {getattr(data_object, 'total_value', 'N/A')}\n\n"
                    holdings_text = holdings_summary + holdings_text
                    
                    holdings_metadata = FilingMetadata(
                        ticker=self.ticker,
                        cik=str(self.cik),
                        company_name=self.company_name,
                        filing_type=filing_type,
                        filing_date=str(filing_date),
                        filing_period=str(filing_period),
                        filing_url=filing_url,
                        section_id="holdings",
                        section_title="Portfolio Holdings",
                        section_level=1,
                        parent_section_title=None,
                        statement_type=None,
                        item_number=None
                    ).dict()
                    
                    processed_chunks.append((holdings_text, holdings_metadata))
                
                # 5. Ownership Reports - use ready-made summaries
                elif filing_type in ["3", "4", "5"]:
                    # Get the ownership summary using the helper method
                    if hasattr(data_object, "get_ownership_summary"):
                        try:
                            ownership_summary = data_object.get_ownership_summary()
                            summary_text = str(ownership_summary)
                            
                            ownership_metadata = FilingMetadata(
                                ticker=self.ticker,
                                cik=str(self.cik),
                                company_name=self.company_name,
                                filing_type=filing_type,
                                filing_date=str(filing_date),
                                filing_period=str(filing_period),
                                filing_url=filing_url,
                                section_id="ownership-summary",
                                section_title="Insider Trading Summary",
                                section_level=1,
                                parent_section_title=None,
                                statement_type=None,
                                item_number=None
                            ).dict()
                            
                            processed_chunks.append((summary_text, ownership_metadata))
                        except Exception as e:
                            logger.warning(f"Error getting ownership summary: {str(e)}")
                            # Fallback to basic owner info will happen automatically
            
            # 6. Process exhibits using attachments.exhibits() directly
            if self.include_exhibits:
                try:
                    for exhibit in filing.attachments.exhibits():
                        try:
                            exhibit_id = exhibit.sequence_number
                            exhibit_type = exhibit.document_type
                            exhibit_description = exhibit.description
                            
                            # Use is_html check for appropriate handling
                            if exhibit.is_html():
                                exhibit_content = exhibit.html()
                                content_type = "html"
                            elif exhibit.is_text():
                                exhibit_content = exhibit.text()
                                content_type = "text"
                            else:
                                continue  # Skip non-text formats
                            
                            # Skip empty exhibits
                            if not exhibit_content or len(exhibit_content.strip()) < 10:
                                continue
                            
                            exhibit_metadata = FilingMetadata(
                                ticker=self.ticker,
                                cik=str(self.cik),
                                company_name=self.company_name,
                                filing_type=filing_type,
                                filing_date=str(filing_date),
                                filing_period=str(filing_period),
                                filing_url=filing_url,
                                section_id=f"exhibit-{exhibit_id}",
                                section_title=f"Exhibit: {exhibit_type}",
                                section_level=1,
                                parent_section_title=None,
                                statement_type=None,
                                item_number=None,
                                content_type=content_type,
                                chunk_type="exhibit"
                            ).dict()
                            
                            # Split into chunks
                            chunks = self.splitter.split_text(exhibit_content)
                            
                            for i, chunk in enumerate(chunks):
                                chunk_metadata = exhibit_metadata.copy()
                                chunk_metadata["chunk_index"] = i
                                chunk_metadata["chunk_count"] = len(chunks)
                                processed_chunks.append((chunk, chunk_metadata))
                        except TimeoutError:
                            # Isolate timeout errors for large exhibits
                            logger.warning(f"Timeout processing exhibit {exhibit.document}")
                        except Exception as e:
                            logger.warning(f"Error processing exhibit {exhibit.document}: {str(e)}")
                except Exception as e:
                    logger.warning(f"Error accessing exhibits: {str(e)}")
            
            # Fallback if no content was extracted
            if not processed_chunks:
                logger.info("No content extracted, using filing text as fallback")
                
                # Get filing text
                filing_text = filing.text()
                
                if filing_text and len(filing_text.strip()) > 10:
                    text_metadata = FilingMetadata(
                        ticker=self.ticker,
                        cik=str(self.cik),
                        company_name=self.company_name,
                        filing_type=filing_type,
                        filing_date=str(filing_date),
                        filing_period=str(filing_period),
                        filing_url=filing_url,
                        section_id="full-text",
                        section_title=f"{filing_type} Full Text",
                        section_level=1,
                        parent_section_title=None,
                        statement_type=None,
                        item_number=None
                    ).dict()
                    
                    # Split text into chunks
                    chunks = self.splitter.split_text(filing_text)
                    
                    for i, chunk in enumerate(chunks):
                        chunk_metadata = text_metadata.copy()
                        chunk_metadata["chunk_index"] = i
                        chunk_metadata["chunk_count"] = len(chunks)
                        processed_chunks.append((chunk, chunk_metadata))
            
            # Mark filing as processed
            self.processed_filings.add(filing_url)
            
            # Add sleep to respect SEC rate limits
            time.sleep(1)  # Sleep for 1 second between API calls
            
            return processed_chunks
        
        except Exception as e:
            logger.error(f"Error processing filing: {str(e)}")
            return []
    
    def add_to_chroma(self, chunks: List[Tuple[str, Dict[str, Any]]]) -> None:
        """Add chunks to ChromaDB."""
        if not chunks:
            return
        
        logger.info(f"Adding {len(chunks)} chunks to ChromaDB")
        
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            texts = [chunk[0] for chunk in batch]
            metadatas = []
            for chunk in batch:
                meta = chunk[1]
                # Replace None values with empty string
                meta_clean = {k: ("" if v is None else v) for k, v in meta.items()}
                metadatas.append(meta_clean)
            ids = []
            for i, (text, metadata) in enumerate(batch):
                key = f"{metadata['ticker']}_{metadata['filing_type']}_{metadata['filing_date']}_{metadata['section_id']}_{metadata.get('chunk_index', 0)}"
                hash_id = hashlib.sha1(f"{key}_{text[:100]}".encode()).hexdigest()
                ids.append(hash_id)
            self.collection.upsert(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added batch of {len(batch)} chunks to ChromaDB")
    
    @staticmethod
    def _process_one_static(args):
        """Static method for ThreadPool to avoid pickling the entire class."""
        processor, filing = args
        return processor.process_one(filing)
    
    def process_one(self, filing):
        """Process a single filing and add to ChromaDB."""
        chunks = self.process_filing(filing)
        self.add_to_chroma(chunks)
        return len(chunks)
    
    def run(self) -> None:
        """Run the SEC filing processor."""
        logger.info(f"Starting SEC filing processing for {self.ticker}")
        
        start_time = time.time()
        
        # Get company info
        self.get_company_info()
        
        # Get filings with pre-loaded objects
        filings = self.get_filings()
        if not filings:
            logger.warning(f"No filings found for {self.ticker}")
            return
        
        logger.info(f"Processing {len(filings)} filings")
        
        # Process filings in parallel using staticmethod
        total_chunks = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(self._process_one_static, (self, filing)) for filing in filings]
            for future in tqdm(as_completed(futures), total=len(filings), desc="Processing filings"):
                chunk_count = future.result()
                total_chunks += chunk_count
        
        # Log summary
        end_time = time.time()
        elapsed = end_time - start_time
        
        logger.info(f"Processing complete:")
        logger.info(f"- Processed {len(filings)} filings")
        logger.info(f"- Added {total_chunks} chunks to ChromaDB")
        logger.info(f"- Total processing time: {elapsed:.2f} seconds")
        logger.info(f"- ChromaDB stored at: {self.output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Build a ChromaDB from SEC filings for a specific company ticker."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Company ticker symbol"
    )
    parser.add_argument(
        "--filing-types",
        type=str,
        default="10-K,10-Q,8-K",
        help="Comma-separated list of filing types to process"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=3,
        help="Number of years of filings to fetch"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./sec_db",
        help="Directory to store ChromaDB"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="FinLang/finance-embeddings-investopedia",
        help="Model to use for embeddings"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of chunks to process in a batch"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing data"
    )
    parser.add_argument(
        "--identity",
        type=str,
        default=None,
        help="Your identity for SEC Edgar API (email address)"
    )
    parser.add_argument(
        "--include-exhibits",
        action="store_true",
        help="Process exhibits in addition to primary filing documents"
    )
    
    args = parser.parse_args()
    
    # Set identity for SEC API
    if args.identity:
        set_identity(args.identity)
    elif "EDGAR_IDENTITY" not in os.environ:
        parser.error("Please provide an identity with --identity or set EDGAR_IDENTITY environment variable")
    
    # Parse filing types
    filing_types = [ft.strip() for ft in args.filing_types.split(",")]
    
    # Create processor
    processor = SECFilingProcessor(
        ticker=args.ticker,
        filing_types=filing_types,
        years=args.years,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
        include_exhibits=args.include_exhibits
    )
    
    # Run processor
    processor.run()


if __name__ == "__main__":
    main()
