import os
import re
import time
from urllib.parse import urlparse

# from papersdb import *
import pandas as pd
import pyautogui
from selenium import webdriver


def convert_ieee_url(landing_url):
    # Define the pattern to match the original URL

    # landing url: https://ieeexplore.ieee.org/document/4443212
    # pdf: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4443212

    pattern = r"https://ieeexplore\.ieee\.org/document/(\d+)"

    # Check if the URL matches the pattern
    match = re.match(pattern, landing_url)

    if match:
        # Extract the document number from the URL
        document_number = match.group(1)

        # Construct the new URL
        pdf_url = f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={document_number}"
        return pdf_url
    else:
        return None


df = pd.read_parquet("db/papers.parquet")

# extract domain
df["domain"] = df["landing_url"].apply(lambda x: urlparse(x).netloc)

# filter by IEEE
domain = "ieeexplore.ieee.org"
_indices = df.index[df["domain"] == domain].tolist()
filtered_df = df.loc[_indices]
print("shape of filtered_df: ", filtered_df.shape)

# sort by influential citations
filtered_df_sorted = filtered_df.sort_values(
    by="influentialCitationCount", ascending=False
)


# Set up Chrome options
chrome_options = webdriver.ChromeOptions()
# chrome_options.add_extension('AlwaysClearDownloads2.crx')

# Initialize the Chrome driver
driver = webdriver.Chrome(options=chrome_options)


print("------ Downloading PDFs")
i = 0
for _index, row in filtered_df_sorted.iterrows():
    i += 1
    print("============", i)
    landing_url = row["landing_url"]
    pdf_url = convert_ieee_url(landing_url)
    if pdf_url:
        print(pdf_url)
        if os.path.exists(f"db/pdfs/{row['paperId']}.pdf"):
            print("file exists!")
        else:
            driver.get(pdf_url)
            time.sleep(30)  # Wait for the PDF to load

            try:
                # Adjust these coordinates based on your screen resolution and browser layout
                pyautogui.moveTo(
                    x=845, y=175, duration=2
                )  # Move to a general location where the download button might be
                pyautogui.click()
                time.sleep(3)

                pyautogui.write(
                    f"C:\\Users\\vansari\\Documents\\PhotonicAI\\PhotonicsAI\\misc-tests\\papers\\db\\pdfs\\{row['paperId']}.pdf"
                )
                time.sleep(1)
                pyautogui.press("enter")
                time.sleep(1)
                pyautogui.moveTo(x=400, y=575, duration=2)
                time.sleep(1)
                pyautogui.click()
                time.sleep(1)

            except Exception as e:
                print(f"Failed to download from {pdf_url}: {str(e)}")

print("====== DONE! ======")

driver.quit()
