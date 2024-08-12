import requests
import pandas as pd
import os
from tqdm import tqdm
from typing import List
import urllib
import regex as re


# train data extraction

train_df = pd.read_csv('csvs/mle-1-assign-dataset - train_data.csv')

print(train_df.describe)

print('Total no.of target categories: ', train_df['target_col'].unique().tolist())
print('Total no.of training data =', len(train_df))

for group_name, grp_df in train_df.groupby(by='target_col'):
    print(group_name, f'-> total no.of occurances in this category: {len(grp_df)}')


# train_df['datasheet_link'].dropna(inplace= True)
# train_df['datasheet_link'].drop('-', inplace= True)
train_data_links = train_df['datasheet_link'].to_list()

unique_train_data_links = list(set(train_data_links))

target_cols = []
for link in unique_train_data_links:
    target_cols.append(train_df.loc[train_df['datasheet_link'] == link, 'target_col'].values[0])

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
    for target,url in zip(target_cols, tqdm(pdf_url_lists, desc="Downloading PDFs", unit="file")):
            
            file_name = url.split('/')[-1]
            print('URL******', url)

            if 'www.te.com/' in url:

                file_name = get_filename_from_te_url(url)
                if file_name is None:
                    file_name = url.split('/')[-1]  
            
            elif url.startswith('//mm'):
                print('url starts with //mm')
                url = f'https:{url}'  
                print('updated URL: ', url)

            elif '&filename' or '&download_file' in url:
                file_name = get_filename_from_url(url)
                if file_name is None:
                   file_name = url.split('/')[-1]   

            elif file_name == '' or url.endswith('/') and '.pdf?' not in file_name:       
                file_name = '_'.join(url.split('/')[-2:])
                if file_name.endswith('_') and not file_name.endswith('.pdf'):
                     file_name = f'{file_name[:-1]}.pdf'
                print('url ends with / so using the new filename: ', file_name)      

            # elif os.path.exists(os.path.join(data_dir,file_name)):
            #     print(f'"{url}" downloaded already')
            # elif os.path.exists(os.path.join(data_dir, f"{file_name}.pdf")):
            #      print(f'url already downloaded at: {os.path.join(data_dir, f"{file_name}.pdf")}')
            

            elif '.pdf?' in file_name:
                file_name  = clean_filename(file_name)
                if file_name == '.pdf':
                    file_name = f"{url.split('/')[-1].split('.pdf?')[0]}.pdf"
                    if file_name.endswith('_'):
                        file_name = f'{file_name[:-1]}.pdf'

            if '.php?' in file_name:
                print('filename doesn"t end with .pdf and its url: ', url)
                nf = get_filename_from_pdf_url(file_name)
                if nf is not None and nf.endswith('.pdf'):
                    file_name = nf
                else:
                    file_name = '_'.join(url.split('/')[-2:])
                    if file_name.endswith('_') or not file_name.endswith('.pdf'):
                      file_name = f'{file_name[:-1]}.pdf'


            elif file_name == '.pdf':
                 file_name = '_'.join(url.split('/')[-2:])
                 if file_name.endswith('_'):
                      file_name = f'{file_name[:-1]}.pdf'

            
            if not file_name.endswith('.pdf'):
                file_name = f'{file_name}.pdf'


            print('\n', f'PDF-URL : {url}')
            print('file name: ', file_name)
            file_path = os.path.join(data_dir, file_name)
            if os.path.exists(file_path.replace('.pdf', f'--{target}.pdf')):
                pass
            else:
                try:
                    response = requests.get(url, headers= headers, allow_redirects= True)
                    if response.status_code == 200:
                        with open(file_path.replace('.pdf', f'--{target}.pdf'), 'wb') as file:
                            file.write(response.content)
                        print(f"Downloaded {file_name} successfully.")
                    else:
                        print(f"Failed to download {url}. Status code: {response.status_code}")
                        failed_urls.append(url)
                except requests.RequestException as error:
                    print(f'Error occurred: {error}')
                    failed_urls.append(url)
                    pass
                target_col_links.append([url, target])
    return target_col_links, failed_urls

target_col_links, failed_urls = download_pdfs(unique_train_data_links, target_cols, 'data/train')


new_tr_df  = pd.DataFrame(target_col_links, columns= ['url', 'target'])

new_tr_df.to_csv('processed_train_data.csv', sep = '|', index= False)