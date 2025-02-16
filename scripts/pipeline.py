import logging
import shutil
import asyncio
import zipfile
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

import aiofiles
import aiohttp
import aioboto3
import pandas as pd
from botocore.exceptions import ClientError

from knmi_fetcher import download_file
from grib_to_parquet import grib_files_to_dataframe

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger("pipeline")

SCRIPTS_PATH = Path(__file__).parent


async def s3_file_exists(s3_bucket: str, key: str) -> bool:
    """Check if a file exists in S3 bucket asynchronously"""
    try:
        async with aioboto3.Session().client("s3") as s3:
            await s3.head_object(Bucket=s3_bucket, Key=key)
            return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise


async def process_single_file(
    session: aiohttp.ClientSession,
    file_info: Dict[str, Any],
    dataset_name: str,
    dataset_version: str,
    output_dir: Path,
    locations_filter: List[int],
    s3_bucket: str = None,
) -> None:
    """Process a single file through the entire pipeline with cleanup"""
    zip_path = unzip_dir = None
    parquet_path = output_dir / f"{file_info['filename'].rsplit('.', 1)[0]}.parquet"

    # Check if output file already exists
    if parquet_path.exists():
        logger.info(f"Skipping {file_info['filename']}: Output file already exists")
        return

    try:
        # Check if output exists in S3
        if s3_bucket and await s3_file_exists(s3_bucket, parquet_path.name):
            logger.info(f"Skipping {file_info['filename']}: {parquet_path.name} exists in S3")
            return

        # 1. Download the file
        zip_path = output_dir / file_info["filename"]
        await download_file(
            session, dataset_name, dataset_version, file_info, output_dir
        )

        # 2. Unzip using thread pool
        unzip_dir = output_dir / "unzipped" / zip_path.stem
        with ThreadPoolExecutor() as pool:
            await asyncio.get_event_loop().run_in_executor(
                pool, lambda: extract_zip(zip_path, unzip_dir)
            )

        # 3. Process GRIB files
        grib_files = sorted(unzip_dir.glob("*_GB"))
        with ThreadPoolExecutor() as pool:
            df = await asyncio.get_event_loop().run_in_executor(
                pool, lambda: grib_files_to_dataframe(grib_files, locations_filter)
            )

        # 4. Save as Parquet
        with ThreadPoolExecutor() as pool:
            await asyncio.get_event_loop().run_in_executor(
                pool, lambda: df.to_parquet(parquet_path)
            )

        # 5. S3 Upload (async)
        if s3_bucket:
            async with (
                aioboto3.Session().client("s3") as s3,
                aiofiles.open(parquet_path, "rb") as f,
            ):
                await s3.upload_fileobj(f, s3_bucket, parquet_path.name)

        logger.info(f"Processed {file_info['filename']} successfully")

    except Exception as e:
        logger.error(f"Failed processing {file_info['filename']}: {e}")
    finally:
        # Cleanup temporary files
        with ThreadPoolExecutor() as pool:
            # Delete zip file
            if zip_path and zip_path.exists():
                await asyncio.get_event_loop().run_in_executor(pool, zip_path.unlink)
            # Delete unzipped directory
            if unzip_dir and unzip_dir.exists():
                await asyncio.get_event_loop().run_in_executor(
                    pool, shutil.rmtree, unzip_dir
                )


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    """Synchronous zip extraction helper"""
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zip_ref:
        zip_ref.extractall(output_dir)


async def process_pipeline(
    api_key: str,
    dataset_name: str,
    dataset_version: str,
    output_dir: Path = Path("../data/processed"),
    max_concurrent: int = 10,
    s3_bucket: str | None = None,
) -> None:
    """Main processing pipeline with concurrency control"""
    # Load configuration

    file_queue = pd.read_json(SCRIPTS_PATH / "file_queue.json").to_dict("records")
    locations_filter = pd.read_csv(SCRIPTS_PATH / "selected_locations.csv")["location_idx"].values
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create processing tasks
    semaphore = asyncio.Semaphore(max_concurrent)
    async with aiohttp.ClientSession(headers={"Authorization": api_key}) as session:
        tasks = []
        for file_info in file_queue:
            await semaphore.acquire()
            task = asyncio.create_task(
                process_single_file(
                    session=session,
                    file_info=file_info,
                    dataset_name=dataset_name,
                    dataset_version=dataset_version,
                    output_dir=output_dir,
                    locations_filter=locations_filter,
                    s3_bucket=s3_bucket,
                )
            )
            task.add_done_callback(lambda _: semaphore.release())
            tasks.append(task)

        await asyncio.gather(*tasks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KNMI Data Processing Pipeline")
    parser.add_argument("-k", "--api_key", required=True)
    parser.add_argument("-d", "--dataset_name", required=True)
    parser.add_argument("-v", "--dataset_version", required=True)
    parser.add_argument(
        "-o", "--output_dir", type=Path, default=Path("../data/processed")
    )
    parser.add_argument("--s3_bucket", type=str, default=None)
    parser.add_argument("-j", "--max_concurrent", type=int, default=1)

    args = parser.parse_args()

    asyncio.run(
        process_pipeline(
            api_key=args.api_key,
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version,
            output_dir=args.output_dir,
            max_concurrent=args.max_concurrent,
            s3_bucket=args.s3_bucket,
        )
    )
