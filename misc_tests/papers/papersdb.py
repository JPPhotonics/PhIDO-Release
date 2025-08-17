import os
import time
from io import BytesIO
from urllib.parse import urlparse

import pandas as pd
import requests
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
from pypdf import PdfReader
from requests.adapters import HTTPAdapter
from selenium import webdriver
from urllib3.util.retry import Retry

load_dotenv()


def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def papers_by_search(query, papers_N=800):
    # max papers_N is 1000
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": os.getenv("SEMANTICSCHOLAR_API_KEY")}

    query_params = {
        "query": query,
        "fieldsOfStudy": "Physics,Engineering",
        "fields": "title,abstract,year,url,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,venue,externalIds,citations,references",
        "minCitationCount": 1,
        "limit": 100,
        "offset": 0,
    }

    papers_retrieved = 0
    papers_list = []
    citations_list = []
    references_list = []

    session = requests_retry_session()

    while papers_retrieved < papers_N:
        try:
            searchResponse = session.get(url, params=query_params, headers=headers)

            if searchResponse.status_code == 200:
                response_data = searchResponse.json()
                papers = response_data.get("data", [])

                for paper in papers:
                    papers_list.append(
                        {
                            "paperId": paper["paperId"],
                            "title": paper.get("title"),
                            "abstract": paper.get("abstract"),
                            "year": paper.get("year"),
                            "url": paper.get("url"),
                            "citationCount": paper.get("citationCount", 0),
                            "influentialCitationCount": paper.get(
                                "influentialCitationCount", 0
                            ),
                            "isOpenAccess": paper.get("isOpenAccess", False),
                            "openAccessPdf": paper.get("openAccessPdf", {}).get("url")
                            if paper.get("openAccessPdf")
                            else None,
                            "venue": paper.get("venue"),
                            "externalIds": paper.get("externalIds"),
                        }
                    )

                    # Collect citations and references separately, only keep the offspring IDs
                    citations_list.extend(
                        [
                            {"citationId": citation["paperId"]}
                            for citation in paper.get("citations", [])
                        ]
                    )
                    references_list.extend(
                        [
                            {"referenceId": reference["paperId"]}
                            for reference in paper.get("references", [])
                        ]
                    )

                papers_retrieved += len(papers)

                if "next" in response_data:
                    query_params["offset"] = response_data["next"]
                else:
                    break

                print(
                    f"Retrieved and stored {papers_retrieved}/{response_data['total']}"
                )
            else:
                print(
                    f"Request failed with status code {searchResponse.status_code}: {searchResponse.text}"
                )
                break

        except requests.exceptions.RequestException as e:
            print(f"Network error occurred: {e}")
            break

        time.sleep(2)  # my current rate limit is 1 request per second

    papers_df = pd.DataFrame(papers_list)
    citations_df = pd.DataFrame(citations_list)
    references_df = pd.DataFrame(references_list)

    return papers_df, citations_df, references_df


def papers_by_id(paper_ids):
    url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    headers = {"x-api-key": os.getenv("SEMANTICSCHOLAR_API_KEY")}

    # Truncate the list to the first 500 items
    paper_ids = paper_ids[:500]
    papers_list = []
    citations_list = []
    references_list = []

    session = requests_retry_session()

    try:
        response = session.post(
            url,
            json={"ids": paper_ids},
            params={
                "fields": "title,abstract,year,url,citationCount,influentialCitationCount,isOpenAccess,openAccessPdf,venue,externalIds,citations,references"
            },
            headers=headers,
        )

        if response.status_code == 200:
            papers = response.json()

            for paper in papers:
                papers_list.append(
                    {
                        "paperId": paper["paperId"],
                        "title": paper.get("title"),
                        "abstract": paper.get("abstract"),
                        "year": paper.get("year"),
                        "url": paper.get("url"),
                        "citationCount": paper.get("citationCount", 0),
                        "influentialCitationCount": paper.get(
                            "influentialCitationCount", 0
                        ),
                        "isOpenAccess": paper.get("isOpenAccess", False),
                        "openAccessPdf": paper.get("openAccessPdf", {}).get("url")
                        if paper.get("openAccessPdf")
                        else None,
                        "venue": paper.get("venue"),
                        "externalIds": paper.get("externalIds"),
                    }
                )

                # Collect citations and references separately, only keep the offspring IDs
                citations_list.extend(
                    [
                        {"citationId": citation["paperId"]}
                        for citation in paper.get("citations", [])
                    ]
                )
                references_list.extend(
                    [
                        {"referenceId": reference["paperId"]}
                        for reference in paper.get("references", [])
                    ]
                )

            print(f"Fetched additional info for {len(paper_ids)} papers.")
        else:
            print(
                f"Request failed with status code {response.status_code}: {response.text}"
            )

        time.sleep(2)  # my current rate limit is 1 request per second

    except requests.exceptions.RequestException as e:
        print(f"Network error occurred: {e}")

    papers_df = pd.DataFrame(papers_list)
    citations_df = pd.DataFrame(citations_list)
    references_df = pd.DataFrame(references_list)

    return papers_df, citations_df, references_df


def recursive_paper_search(queries, depth=5, papers_N=800):
    # Initialize an empty DataFrame to store all results
    final_papers_df = pd.DataFrame()

    def search_and_expand(paper_ids, current_depth, query):
        time.sleep(5)
        if current_depth > depth:
            return

        if current_depth == 0:
            # Initial search by query
            papers_df, citations_df, references_df = papers_by_search(
                query, papers_N=papers_N
            )
        else:
            print(f"Depth {current_depth}: Expanding {len(paper_ids)} papers.")
            # Search by paper IDs
            papers_df, citations_df, references_df = papers_by_id(paper_ids)

        # Add a column indicating the depth level and the query used
        papers_df["depth"] = current_depth
        papers_df["query"] = query

        # Append only new rows where 'paperId' does not exist in the final DataFrame
        nonlocal final_papers_df
        print("shape of new papers from query:", papers_df.shape, query)
        final_papers_df = pd.concat([final_papers_df, papers_df]).drop_duplicates(
            subset=["paperId"]
        )
        print("shape of final df after deduplication:", final_papers_df.shape)
        print("-------")

        # Extract unique citation and reference IDs for the next depth level
        citations_ids = (
            citations_df["citationId"]
            if "citationId" in citations_df.columns
            else pd.Series()
        )
        references_ids = (
            references_df["referenceId"]
            if "referenceId" in references_df.columns
            else pd.Series()
        )

        # Concatenate the series and get unique IDs as a list
        next_paper_ids = pd.concat([citations_ids, references_ids]).unique().tolist()

        # Recursively search for the next depth level
        search_and_expand(next_paper_ids, current_depth + 1, query)

    # Iterate through each query in the list
    for query in queries:
        # Start the recursive search for each query
        search_and_expand([], 0, query)

    # Return the final DataFrame containing all unique papers
    return final_papers_df


def download_pdf_requests(pdf_url, target_dir, paperid):
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    try:
        # Send a GET request to the PDF URL
        response = requests.get(pdf_url)
        response.raise_for_status()  # Raise an error for bad status codes

        # Validate the PDF content before saving
        if is_pdf_valid(response.content):
            # Extract the filename from the URL
            pdf_filename = f"{paperid}.pdf"
            pdf_filepath = os.path.join(target_dir, pdf_filename)

            # Write the PDF content to a file
            with open(pdf_filepath, "wb") as pdf_file:
                pdf_file.write(response.content)

            print(f"PDF downloaded and validated successfully: {pdf_filepath}")
            return pdf_filepath
        else:
            print("The PDF content is invalid and will not be saved.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Failed to download PDF using requests: {e}")
        return None


def is_pdf_valid(pdf_content):
    try:
        reader = PdfReader(BytesIO(pdf_content))
        # Attempt to access the first page or metadata
        reader.pages[0]
        return True
    except Exception as e:
        # print(f"Invalid PDF content: {e}")
        _error = e
        return False


def download_pdf_selenium(pdf_url, target_dir, paperid):
    # not very effective. cannot handle cloudflare challenges

    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Set up Firefox options
    options = webdriver.FirefoxOptions()
    options.add_argument("--headless")

    try:
        # Initialize the Firefox WebDriver
        driver = webdriver.Firefox(options=options)

        # Navigate to the PDF URL
        driver.get(pdf_url)

        # Extract the filename from the URL
        pdf_filename = f"{paperid}.pdf"
        pdf_filepath = os.path.join(target_dir, pdf_filename)

        # Save the PDF content (you may need to handle this differently based on the page)
        with open(pdf_filepath, "wb") as pdf_file:
            pdf_file.write(
                driver.page_source.encode("utf-8")
            )  # This may need adjustment

        print(f"PDF downloaded successfully (selenium): {pdf_filepath}")
        return pdf_filepath

    except Exception as e:
        print(f"Failed to download PDF using Selenium: {e}")
        return None

    finally:
        # Close the browser
        driver.quit()


def download_pdf_with_fallback(pdf_url, target_dir, paperid):
    # Try downloading with requests first
    pdf_path = download_pdf_requests(pdf_url, target_dir, paperid)

    if pdf_path:
        return True
    else:
        return False

    # If requests fails, try downloading with Selenium
    # pdf_path = download_pdf_selenium(pdf_url, target_dir, paperid)

    # if pdf_path:
    #     return True
    # else:
    #     return False


def save_webpage(url, base_filename, wait_until="networkidle", wait_for_selector=None):
    try:
        with sync_playwright() as p:
            # Launch a browser (use 'chromium', 'firefox', or 'webkit')
            browser = p.chromium.launch()
            page = browser.new_page()

            # Navigate to the URL
            try:
                page.goto(url, wait_until=wait_until)
            except Exception as e:
                print(f"Failed to navigate to {url}: {e}")
                browser.close()
                return None, None

            # Scroll to the bottom of the page
            page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
            page.wait_for_timeout(3000)

            # Optionally wait for a specific selector to ensure the page is fully loaded
            if wait_for_selector:
                try:
                    page.wait_for_selector(wait_for_selector)
                except Exception as e:
                    print(
                        f"Timeout while waiting for selector {wait_for_selector}: {e}"
                    )
                    browser.close()
                    return None, None

            # Get the page content (including HTML and potentially inlined resources)
            try:
                html_content = page.content()
            except Exception as e:
                print(f"Failed to get content from {url}: {e}")
                browser.close()
                return None, None

            # Get the final landing URL after any redirections
            final_url = page.url

            # Define file paths for HTML and PDF
            html_file_path = f"{base_filename}.html"
            pdf_file_path = f"{base_filename}.pdf"

            # Save the HTML content to a file
            try:
                with open(html_file_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
            except OSError as e:
                print(f"Failed to save HTML content to {html_file_path}: {e}")
                browser.close()
                return None, None

            # Save the PDF version of the page
            try:
                page.pdf(path=pdf_file_path)
            except Exception as e:
                print(f"Failed to save PDF content to {pdf_file_path}: {e}")
                browser.close()
                return None, None

            # Close the browser
            browser.close()

            return html_content, final_url

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None


# async def save_page_as_html(url, file_name, wait_until='networkidle', wait_for_selector=None):
#     try:
#         async with async_playwright() as p:
#             # Launch a browser (use 'chromium', 'firefox', or 'webkit')
#             print('0')
#             browser = await p.chromium.launch()
#             page = await browser.new_page()

#             # Navigate to the URL
#             print('1')
#             try:
#                 await page.goto(url, wait_until=wait_until)
#             except PlaywrightTimeoutError as e:
#                 print(f"Failed to navigate to {url} within the timeout period: {e}")
#                 await browser.close()
#                 return None, None
#             except PlaywrightError as e:
#                 print(f"Failed to navigate to {url}: {e}")
#                 await browser.close()
#                 return None, None

#             # Optionally wait for a specific selector to ensure the page is fully loaded
#             if wait_for_selector:
#                 try:
#                     await page.wait_for_selector(wait_for_selector)
#                 except PlaywrightTimeoutError as e:
#                     print(f"Timeout while waiting for selector {wait_for_selector}: {e}")
#                     await browser.close()
#                     return None, None

#             # Get the page content (including HTML and potentially inlined resources)
#             try:
#                 html_content = await page.content()
#             except PlaywrightError as e:
#                 print(f"Failed to get content from {url}: {e}")
#                 await browser.close()
#                 return None, None

#             # Get the final landing URL after any redirections
#             final_url = page.url

#             # Save the HTML content to a file
#             try:
#                 with open(file_name, 'w', encoding='utf-8') as f:
#                     f.write(html_content)
#             except IOError as e:
#                 print(f"Failed to save HTML content to {file_name}: {e}")
#                 await browser.close()
#                 return None, None

#             # Close the browser
#             await browser.close()

#             return html_content, final_url

#     except PlaywrightError as e:
#         print(f"Playwright encountered an error: {e}")
#         return None, None
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         return None, None


def is_valid_url(url):
    try:
        result = urlparse(url)
        # A valid URL must have a scheme (http, https, etc.) and a network location (domain, IP, etc.)
        return all([result.scheme, result.netloc])
    except ValueError:
        print("Invalid URL:", url)
        return False


# if 0:
#     import asyncio
#     from PhotonicsAI.Photon import llm_api
#     html_content = asyncio.run(save_page_as_html('https://www.doi.org/10.1117/12.2255794', 'example.html'))
#     sys_prompt = 'This is a webpage of a scientific article. Can you find a PDF download link? only return the link with no other information.'
#     r = llm_api.call_llm(html_content, sys_prompt, llm_api_selection='gpt-4o-mini')
#     print(r)


# Example usage
# all_papers_df = recursive_paper_search('integrated silicon photonics switch networks', depth=0, papers_N=200)

# print(all_papers_df.head())
# print(all_papers_df.info())

# all_papers_df.to_parquet('papers.parquet', index=False)
# # all_papers_df.to_csv('papers.csv', index=False)
