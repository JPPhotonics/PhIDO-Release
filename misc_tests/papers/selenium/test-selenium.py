import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Data provided directly in the script
data = [
    {"file_name": "file1", "url": "https://www.doi.org/10.1109/ISCAS.2013.6572199"},
    {"file_name": "file2", "url": "https://www.doi.org/10.1117/12.262446"},
    {"file_name": "file3", "url": "https://www.doi.org/10.1364/ipr.2002.ifb1"},
]

# Set up Chrome options to save as MHTML
chrome_options = Options()
# chrome_options.add_argument('--headless')  # Optional: Run in headless mode
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--save-page-as-mhtml")

# Initialize the WebDriver with options
driver = webdriver.Chrome(options=chrome_options)

# Iterate over the data
for entry in data:
    url = entry["url"]
    file_name = entry["file_name"]

    # Open URL
    driver.get(url)
    time.sleep(3)  # Allow time for the page to load

    # Handle cookie warnings
    try:
        accept_button = driver.find_element_by_xpath("//*[contains(text(), 'Accept')]")
        accept_button.click()
        time.sleep(2)  # Allow time for the action to complete
    except:
        pass  # If no cookie warning, continue

    # Save the page as MHTML
    with open(f"{file_name}.mhtml", "w", encoding="utf-8") as f:
        f.write(driver.page_source)

# Close the browser
driver.quit()
