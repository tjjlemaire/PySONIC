# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2023-05-17 15:37:50
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2023-05-18 15:07:34

''' Set of utility functions to download lookup files from remote repository. '''

import logging
import requests
import os
import pandas as pd
from tqdm import tqdm

from .utils import logger, LOOKUP_DIR

logger.setLevel(logging.INFO)

# Github API URL to fetch the content of lookup folder
LOOKUP_REMOTE_API_FOLDER = 'https://api.github.com/repos/tjjlemaire/PySONIC/contents/PySONIC/lookups'


def extract_lookup_info(item):
    ''' Extract lookup file information from a Github API item '''
    target_url = item['html_url'].replace('blob', 'raw')
    size = get_file_info(target_url, head_only=False)['Content-Length']
    return {'name': item['name'], 'url': target_url, 'size (MB)': int(size) / (1024**2)}


def list_remote_lookups():
    ''' List all lookup files present in the remote lookup folder '''
    # Fetch remote lookup folder content
    logger.info(f'fetching lookups list from "{LOOKUP_REMOTE_API_FOLDER}"')
    res = requests.get(LOOKUP_REMOTE_API_FOLDER)

    # Check response status, and raise error if not 200
    if res.status_code != 200:
        raise ValueError(res.json()['message'])
    items = res.json()
    
    # Filter out non-files
    files = [item for item in items if item['type'] == 'file']

    # Filter out non-lookup files
    lkpfiles = [item for item in files if item['name'].endswith('.pkl')]

    # Extract file information (download URL and size) for each lookup file, and
    # store in dataframe
    lkpsdf = pd.DataFrame([pd.Series(extract_lookup_info(item)) for item in lkpfiles])

    # Compute total memory size
    totsize = lkpsdf['size (MB)'].sum()

    # Log and return
    logger.info(f'found {len(lkpsdf)} lookup files:\n{lkpsdf}\ntotal size: {totsize:.1f} MB')
    return lkpsdf


def get_file_info(url, head_only=True):
    '''
    Get information about a remote file
    
    :param url: URL to remote file
    :return: dictionary of file settings
    '''
    # Fetch remote file information
    logger.info(f'fetching file info from "{url}"')
    if head_only:
        res = requests.head(url)
    else:
        res = requests.get(url)
    
    # Check response status, and raise error if not 200
    if res.status_code != 200:
        errmsg = res.json()['message']
        raise ValueError(errmsg)

    # Return info headers as dictionary
    return res.headers


def download_file(url, fname=None, local_dir=None, overwrite=True, chunk_size=1024):
    ''' 
    Download file from url.
    
    :param url: URL to remote file
    :param fname (optional): file name to save to. If no file name is given, a filename is inferred from the URL.
    :param local_dir (optional): local directory to save to. If no directory is given, the parent directory of this module is used.
    :param overwrite (optional): whether to overwrite existing local file (defaults to True)
    :param chunk_size: chunk size for download
    :return: path to local downloaded file
    '''
    # Extract file name from URL if not given
    if fname is None:
        fname = url.split('/')[-1]

    # Get path to local directory, if not given
    if local_dir is None:
        local_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check that local directory exists
    if not os.path.isdir(local_dir):
        raise ValueError(f'Invalid local directory: {local_dir}')
    
    # Assemble path to local file
    fpath = os.path.join(local_dir, fname)

    # If local file already exists, check overwrite flag
    if os.path.isfile(fpath):
        # If overwrite flag is not set, raise error
        if not overwrite:
            raise FileExistsError(f'"{fpath}" alreay exists')
        # If overwrite flag is set, remove existing file
        else:
            logger.warning(f'Removing existing file: {fpath}')
            os.remove(fpath)

    # Log download
    logger.info(f'Downloading "{fname}" from "{url}" to "{local_dir}"')

    # Request information about remote file
    res = requests.get(url, stream=True)

    # Check response status, and raise error if not 200
    if res.status_code != 200:
        raise ValueError(res.json()['message'])

    # Extract content length from response headers
    total = int(res.headers.get('content-length', 0))

    # Download file into appropriate destination
    with open(fpath, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in res.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)
    
    # Return path to local file
    return fpath


def download_lookups(**kwargs):
    ''' Download all available lookup files from remote folder '''
    # Get dictionary of remote lookup files
    lkpsdf = list_remote_lookups()

    # Create path to local lookups directory, if not existing
    if LOOKUP_DIR is not None:
        os.makedirs(LOOKUP_DIR, exist_ok=True)

    # For each lookup file
    for _, row in lkpsdf.iterrows():
        # Download from URL, or log warning if error was raised
        try:
            download_file(row['url'], fname=row['name'], local_dir=LOOKUP_DIR, **kwargs)
        except FileExistsError as err:
            logger.warning(f'{err} -> skipping')

    # Return list of downloaded files
    logger.info('downloads completed.')
