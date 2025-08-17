import time

import pyautogui

# Step 1: Open Chrome
pyautogui.press("win")  # Press the Windows key to open the Start menu
time.sleep(1)
pyautogui.write("chrome")  # Type 'chrome' to search for Google Chrome
time.sleep(1)
pyautogui.press("enter")  # Press Enter to open Chrome
time.sleep(3)  # Wait for Chrome to open

# Step 2: Navigate to a webpage
pyautogui.write("https://www.example.com")  # Type the URL of the webpage
pyautogui.press("enter")  # Press Enter to navigate
time.sleep(5)  # Wait for the page to load

# Step 3: Save the webpage
pyautogui.hotkey("ctrl", "s")  # Press Ctrl+S to open the Save dialog
time.sleep(2)
pyautogui.write("saved_page")  # Enter the filename for saving
time.sleep(1)
pyautogui.press("enter")  # Press Enter to save the page
