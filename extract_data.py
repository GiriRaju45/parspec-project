import requests
import pandas as pd
import os
from tqdm import tqdm
from typing import List
import urllib
import regex as re
# from langchain.document_loaders import PyPDFLoader

# train data extraction

test_df = pd.read_csv('csvs/mle-1-assign-dataset - test_data.csv')

print(test_df.describe)

print('Total no.of target categories: ', test_df['target_col'].unique().tolist())
print('Total no.of training data =', len(test_df))

for group_name, grp_df in test_df.groupby(by='target_col'):
    print(group_name, f'-> total no.of occurances in this category: {len(grp_df)}')


# test_df['datasheet_link'].dropna(inplace= True)
# test_df['datasheet_link'].drop('-', inplace= True)
train_data_links = test_df['datasheet_link'].to_list()

unique_train_data_links = list(set(train_data_links))

target_cols = []
for link in unique_train_data_links:
    target_cols.append(test_df.loc[test_df['datasheet_link'] == link, 'target_col'].values[0])

def clean_filename(url):
    # Parse the URL
    parsed_url = urllib.parse.urlparse(url)
    
    # Extract the path component
    path = parsed_url.path
    
    # Extract the filename from the path
    filename = path.split('/')[-1]
    
    # Decode any percent-encoded characters
    filename = urllib.parse.unquote(filename)
    
    # Clean the filename: remove unwanted characters and extra spaces
    filename = re.sub(r'[^\w\s.-]', '', filename)  # Remove special characters
    filename = re.sub(r'\s+', ' ', filename)       # Replace multiple spaces with a single space
    filename = filename.strip()                    # Remove leading and trailing spaces
    
    return filename

from urllib.parse import urlparse, parse_qs

def get_filename_from_pdf_url(url):
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Extract query parameters
    query_params = parse_qs(parsed_url.query)
    
    # Get the 'OutoftheBoxpath' parameter
    path_param = query_params.get('OutoftheBoxpath', [None])[0]
    
    # Extract the filename from the path
    if path_param:
        filename = path_param.split('/')[-1]
        return filename
    return None

def get_filename_from_te_url(url):
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Extract query parameters
    query_params = parse_qs(parsed_url.query)
    
    # Get the 'DocNm' parameter
    doc_name = query_params.get('DocNm', [None])[0]
    
    # Use the 'DocNm' value as the filename, ensuring a .pdf extension
    if doc_name:
        filename = f'{doc_name}.pdf'
        return filename
    return None

def get_filename_from_url(url):
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Extract query parameters
    query_params = parse_qs(parsed_url.query)
    
    # Get the 'filename' parameter
    if 'filename' in url:
        filename = query_params.get('filename', [None])[0]
        return filename
    elif 'download_file' in url:
        filename = query_params.get('download_file', [None])[0]
        return filename
    return None

def download_pdfs( pdf_url_lists : List[str], target_cols, data_dir : str):

    ''' This function takes the list of pdf source links and 
    data directory/folder where it downloads the pdfs'''

    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
    
    failed_urls = [] # list to save the failed urls
    target_col_links = []
    for idx,(target,url) in enumerate(zip(target_cols, tqdm(pdf_url_lists, desc="Downloading PDFs", unit="file"))):
            

            # if url.startswith('//mm'):
            #     print('url starts with //mm')
            #     url = f'https:{url}'  
            #     print('updated URL: ', url)
            file_name = f'{target}_{idx}.pdf'
            print('\n', f'PDF-URL : {url}')
            print('file name: ', file_name)

            file_path = os.path.join(data_dir, file_name)
           

            if not os.path.exists(file_path): # os.path.exists(file_path) or 
                file_name = f'{target}_{idx}.pdf'
                print('\n', f'PDF-URL : {url}')
                print('file name: ', file_name)
                file_path = os.path.join(data_dir, file_name)
               
                # print(f'{file_name} already exists downloaded from the {url} at {file_path}')
                try:
                    response = requests.get(url, headers= headers, allow_redirects= True, timeout= 15)
                    if response.status_code == 200:
                        with open(file_path, 'wb') as file:
                            file.write(response.content)
                        print(f"Downloaded {file_name} successfully.")
                        # loader = PyPDFLoader(file_path)
                        # pages = loader.load_and_split()
                        # for page in pages:
                        #     text += page.page_content
                        # with open(file_path.replace('train_simple', 'train_texts').replace('.pdf', '.txt'), 'wb') as f:
                        #     f.write(text.encode('utf-8'))

                        target_col_links.append([url, file_name, target])
                    else:
                        print(f"Failed to download {url}. Status code: {response.status_code}")
                        failed_urls.append(url)
                except requests.RequestException as error:
                    print(f'Error occurred: {error}')
                    failed_urls.append(url)
            else:
                pass

                # try:
                #     response = requests.get(url, headers= headers, allow_redirects= True)
                #     if response.status_code == 200:
                #         text = ''
                #         with open(file_path, 'wb') as file:
                #             file.write(response.content)
                #         print(f"Downloaded {file_name} successfully. \n extracting text and writing it into txt file..")
                #         # loader = PyPDFLoader(file_path)
                #         # pages = loader.load_and_split()
                #         # for page in pages:
                #         #     text += page.page_content
                #         # with open(file_path.replace('train_simple', 'train_texts').replace('.pdf', '.txt'), 'wb') as f:
                #         #     f.write(text.encode('utf-8'))

                #         target_col_links.append([url, file_name, target])
                #     else:
                #         print(f"Failed to download {url}. Status code: {response.status_code}")
                #         failed_urls.append(url)
                # except requests.RequestException as error:
                #     print(f'Error occurred: {error}')
                #     failed_urls.append(url)
                #     pass
                # # target_col_links.append([url, target])
    return target_col_links, failed_urls

target_col_links, failed_urls = download_pdfs(unique_train_data_links, target_cols, 'data/test')


new_tr_df  = pd.DataFrame(target_col_links, columns= ['url', 'file_name', 'target'])

new_tr_df.to_csv('processed_test_data.csv', sep = '|', index= False)