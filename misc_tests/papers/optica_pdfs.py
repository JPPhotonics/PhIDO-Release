import os
import time
from urllib.parse import urlparse

# from papersdb import *
import pandas as pd
import pyautogui
from selenium import webdriver

download_pdf_button_optica = "download_pdf_button_optica.png"
download_pdf_button_chrome = "download_pdf_button_chrome.png"


df = pd.read_parquet("db/papers.parquet")

# extract domain
df["domain"] = df["landing_url"].apply(lambda x: urlparse(x).netloc)

# filter by IEEE
domain = "opg.optica.org"
_indices = df.index[df["domain"] == domain].tolist()
filtered_df = df.loc[_indices]
print("shape of filtered_df: ", filtered_df.shape)

# sort by influential citations
filtered_df_sorted = filtered_df.sort_values(
    by="influentialCitationCount", ascending=False
)


# Set up Chrome options
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--start-maximized")

# Initialize the Chrome driver
driver = webdriver.Chrome(options=chrome_options)


print("------ Downloading PDFs")
i = 0
for _index, row in filtered_df_sorted.iterrows():
    i += 1
    print("============", i)
    landing_url = row["landing_url"]

    if os.path.exists(f"db/pdfs/{row['paperId']}.pdf"):
        print("file exists!")
    else:
        driver.get(landing_url)
        time.sleep(20)  # Wait for the PDF to load

        try:
            location = pyautogui.locateOnScreen(
                download_pdf_button_optica, confidence=0.8
            )
            center_x, center_y = pyautogui.center(location)
            pyautogui.moveTo(center_x, center_y, duration=2)
            pyautogui.click()
            time.sleep(5 * 60)

            location = pyautogui.locateOnScreen(
                download_pdf_button_chrome, confidence=0.8
            )
            center_x, center_y = pyautogui.center(location)
            pyautogui.moveTo(center_x, center_y - 10, duration=2)
            pyautogui.click()
            time.sleep(3)

            pyautogui.write(
                f"C:\\Users\\vansari\\Documents\\PhotonicAI\\PhotonicsAI\\misc-tests\\papers\\db\\pdfs\\{row['paperId']}.pdf"
            )
            time.sleep(1)
            pyautogui.press("enter")
            time.sleep(2)
            # pyautogui.moveTo(x=400, y=575, duration=2)
            # time.sleep(1)
            # pyautogui.click()
            # time.sleep(1)
            pyautogui.hotkey("ctrl", "w")
            time.sleep(1)

        except Exception as e:
            print(f"Failed to download from {landing_url}: {str(e)}")

print("====== DONE! ======")

driver.quit()
