import argparse

import aiofiles
import aiohttp
import asyncio
import logging
import os
from pathlib import Path
from typing import Any, List, Dict

import pandas as pd
from tqdm import tqdm

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", logging.INFO))

BASE_URL = "https://api.dataplatform.knmi.nl/open-data"


async def list_dataset_files(
    session: aiohttp.ClientSession,
    dataset_name: str,
    dataset_version: str,
    max_keys: int = 500,
) -> List[Dict[str, Any]]:
    """Async function to list all dataset files with pagination"""
    files = []
    next_page_token = None

    while True:
        params = {"maxKeys": str(max_keys)}
        if next_page_token:
            params["nextPageToken"] = next_page_token

        url = f"{BASE_URL}/datasets/{dataset_name}/versions/{dataset_version}/files"

        try:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                files.extend(data.get("files", []))
                next_page_token = data.get("nextPageToken")

                if not next_page_token:
                    break

        except Exception as e:
            logger.error(f"Error listing files: {e}")
            raise

    return files


async def download_file(
    session: aiohttp.ClientSession,
    dataset_name: str,
    dataset_version: str,
    file_info: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Async function to download a single file"""
    try:
        # Get presigned URL
        url = f"{BASE_URL}/datasets/{dataset_name}/versions/{dataset_version}/files/{file_info['filename']}/url"

        async with session.get(url) as response:
            response.raise_for_status()
            download_url = (await response.json())["temporaryDownloadUrl"]

        # Download actual file
        output_path = output_dir / file_info["filename"]

        async with (
            aiohttp.ClientSession() as anonymous_session,
            anonymous_session.get(download_url) as response,
            aiofiles.open(output_path, mode="wb") as f,
        ):
            response.raise_for_status()
            total_size = int(response.headers.get("Content-Length", 0))
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=file_info["filename"],
            ) as progress_bar:
                async for chunk in response.content.iter_chunked(1024):
                    await f.write(chunk)
                    progress_bar.update(len(chunk))

        logger.info(f"Downloaded {file_info['filename']}")

    except Exception as e:
        logger.error(f"Failed to download {file_info['filename']}: {e}")
        raise


async def main():
    parser = argparse.ArgumentParser(description="KNMI Data Platform Downloader")
    parser.add_argument("-k", "--api_key", type=str, required=True)
    parser.add_argument("-d", "--dataset_name", type=str, required=True)
    parser.add_argument("-v", "--dataset_version", type=str, required=True)
    parser.add_argument(
        "-o", "--output_dir", type=Path, default=Path("../data/downloads")
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession(
        headers={"Authorization": args.api_key}
    ) as session:
        # Get all files
        files = await list_dataset_files(
            session, args.dataset_name, args.dataset_version
        )
        logger.info(f"Found {len(files)} files to download")

        pd.DataFrame(files).to_json(args.output_dir / "file_queue.json", index=False)


if __name__ == "__main__":
    asyncio.run(main())
