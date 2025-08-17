import time

import pyautogui
from selenium import webdriver

# Set up Chrome options
chrome_options = webdriver.ChromeOptions()
# chrome_options.add_extension('AlwaysClearDownloads2.crx')

# Initialize the Chrome driver
driver = webdriver.Chrome(options=chrome_options)

# List of URLs to download
urls = [
    "https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4443212",
    "https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9658267",
    "https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8794827",
    # Add more URLs here
]

i = 0
for url in urls:
    driver.get(url)
    time.sleep(60)  # Wait for the PDF to load

    try:
        # Adjust these coordinates based on your screen resolution and browser layout
        pyautogui.moveTo(
            x=845, y=175, duration=2
        )  # Move to a general location where the download button might be
        pyautogui.click()
        time.sleep(5)

        pyautogui.write(
            f"C:\\Users\\vansari\\Documents\\PhotonicAI\\PhotonicsAI\\misc-tests\\papers\\selenium\\downloads\\filename{i}.pdf"
        )
        time.sleep(1)
        pyautogui.press("enter")
        time.sleep(1)
        pyautogui.moveTo(x=400, y=575, duration=2)
        time.sleep(1)
        pyautogui.click()
        time.sleep(1)

        # Construct the expected file name
        # file_name = f"{url.split('arnumber=')[-1]}.pdf"
        # download_path = "/downloads"

        # # Wait until the file is downloaded
        # while not os.path.exists(os.path.join(download_path, file_name)):
        #     time.sleep(1)

        # print(f"Downloaded: {file_name}")

    except Exception as e:
        print(f"Failed to download from {url}: {str(e)}")
    i += 1

driver.quit()
