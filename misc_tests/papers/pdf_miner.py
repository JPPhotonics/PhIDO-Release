import os
import random

# from urllib.parse import urlparse
import time

# from papersdb import *
import pandas as pd
import pyautogui
from selenium import webdriver


def find_button_and_click(button_img_path):
    try:
        location = pyautogui.locateOnScreen(button_img_path, confidence=0.8)
        center_x, center_y = pyautogui.center(location)
        pyautogui.moveTo(center_x, center_y, duration=1)
        pyautogui.click()
    except pyautogui.ImageNotFoundException:
        print("ImageNotFoundException: button not found!", button_img_path)
        return "button not found!"


def find_pdf_link_right_click_save(button_img_path, paperId):
    try:
        location = pyautogui.locateOnScreen(button_img_path, confidence=0.8)
        center_x, center_y = pyautogui.center(location)
        pyautogui.moveTo(center_x, center_y, duration=2)
        pyautogui.rightClick()
        time.sleep(1)
        find_button_and_click("img/chrome_save_link_as.png")
        time.sleep(10)
        pyautogui.write(
            f"C:\\Users\\vansari\\Documents\\PhotonicAI\\PhotonicsAI\\misc-tests\\papers\\db\\pdfs\\{paperId}.pdf"
        )
        time.sleep(1)
        pyautogui.press("enter")
        time.sleep(1)
        # pyautogui.moveTo(100, 500, duration=1)
    except pyautogui.ImageNotFoundException:
        print("ImageNotFoundException: button not found!", button_img_path)
        return "button not found!"


def find_save_chrome_pdf(paperId):
    try:
        find_button_and_click("img/chrome_download_pdf_button.png")
        time.sleep(10)
        pyautogui.write(
            f"C:\\Users\\vansari\\Documents\\PhotonicAI\\PhotonicsAI\\misc-tests\\papers\\db\\pdfs\\{paperId}.pdf"
        )
        time.sleep(1)
        pyautogui.press("enter")
        time.sleep(2)
        pyautogui.press("esc")
        time.sleep(2)
    except pyautogui.ImageNotFoundException:
        print("ImageNotFoundException: Chrome button not found!")


def optica_traffic_error():
    try:
        pyautogui.locateOnScreen("img/optica_error.png", confidence=0.8)
        time.sleep(random.randint(5 * 60, 20 * 60))
        # driver.refresh()
        pyautogui.hotkey("ctrl", "r")
        time.sleep(5)
        return True
    except:
        return False


def optica_bot(paperId):
    time.sleep(random.randint(1 * 60, 10 * 60))

    r = True
    while r:
        r = optica_traffic_error()

    find_button_and_click("img/optica_download_pdf_button.png")

    r = True
    while r:
        r = optica_traffic_error()

    time.sleep(5)

    find_save_chrome_pdf(paperId)
    time.sleep(3)
    pyautogui.hotkey("ctrl", "w")
    time.sleep(2)


def ieee_bot(paperId):
    time.sleep(5)
    find_button_and_click("img/ieee_download_pdf_button.png")
    time.sleep(40)

    # ieee opens pdf in the same tab
    find_save_chrome_pdf(paperId)
    time.sleep(2)


def spie_bot(paperId):
    time.sleep(30)
    find_pdf_link_right_click_save("img/spie_download_pdf_button.png", paperId)
    time.sleep(5)


def springer_bot(paperId):
    time.sleep(5)
    find_button_and_click("img/springer_cookies.png")
    time.sleep(30)

    # springer directly opens pdf
    r = find_pdf_link_right_click_save("img/springer_download_pdf_button1.png", paperId)
    if r == "button not found!":
        find_pdf_link_right_click_save("img/springer_download_pdf_button2.png", paperId)

    time.sleep(5)


def wiley_bot(paperId):
    # doesn't work
    # has a very persistent cloudflare human detector
    # it shows up randomly

    time.sleep(5)

    find_button_and_click("img/wiley_download_pdf_button1.png")
    time.sleep(30)
    find_pdf_link_right_click_save("img/wiley_download_pdf_button2.png", paperId)

    time.sleep(5)


def aip_bot(paperId):
    time.sleep(5)
    find_button_and_click("img/aip_cookies.png")
    time.sleep(30)
    find_pdf_link_right_click_save("img/aip_download_pdf_button.png", paperId)
    time.sleep(5)


def acm_bot(paperId):
    time.sleep(30)
    find_pdf_link_right_click_save("img/acm_download_pdf_button.png", paperId)
    time.sleep(5)


def iop_bot(paperId):
    time.sleep(5)
    find_button_and_click("img/iop_cookies.png")
    time.sleep(5)
    r = find_button_and_click("img/iop_download_pdf_button.png")
    if r is None:
        time.sleep(40)
        find_save_chrome_pdf(paperId)
        time.sleep(2)
        pyautogui.hotkey("ctrl", "w")
        time.sleep(2)


def nature_bot(paperId):
    time.sleep(5)
    find_button_and_click("img/nature_cookies.png")
    time.sleep(30)
    find_pdf_link_right_click_save("img/nature_download_pdf_button.png", paperId)
    time.sleep(5)


def cleanup():
    try:
        location = pyautogui.locateOnScreen("acrobat.png", confidence=0.8)
        center_x, center_y = pyautogui.center(location)
        pyautogui.moveTo(center_x - 41, center_y - 22, duration=1)
        pyautogui.click()
    except:
        pass

    pyautogui.moveTo(100, 500, duration=1)
    time.sleep(2)
    driver.switch_to.window(driver.window_handles[0])
    time.sleep(2)
    pyautogui.press("esc")
    time.sleep(2)


def check_file_exist(paperId):
    file_exist = False
    pdf_dirs = [
        "db/pdfs/",
    ]
    for _dir in pdf_dirs:
        if os.path.exists(f"{_dir}{paperId}.pdf"):
            file_exist = True
    return file_exist


df = pd.read_parquet("db/papers_wdm.parquet")
# df = pd.read_parquet('db/papers_switch.parquet')
if "landing_url" not in df.columns:
    df["landing_url"] = ""

df = df.sample(frac=1).reset_index(drop=True)
# df.sort_values(by='influentialCitationCount', ascending=False, inplace=True)

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=chrome_options)


# i = j = 0
# if 1:
#     for index, row in df.iterrows():
#         file_exist = check_file_exist(row['paperId'])
#         if file_exist:
#             i+=1

# print(df.info())
# print('+++++++++++', i)


if 0:  # Downloading open access PDFs
    for index, row in df.iterrows():
        pdf_url = row["openAccessPdf"]
        file_exist = check_file_exist(row["paperId"])
        if (not file_exist) & (pdf_url is not None):
            try:
                driver.get(pdf_url)
                time.sleep(20)
                find_save_chrome_pdf(row["paperId"])
                time.sleep(2)
            except:
                pass
            cleanup()


if 1:
    for index, row in df.iterrows():
        doiurl = row["externalIds"]["DOI"]
        doi_exist = doiurl is not None
        file_exist = check_file_exist(row["paperId"])

        if (not file_exist) & (doi_exist):
            # if pd.isna(row['landing_url']):
            if row["landing_url"] == "":
                try:
                    driver.get("https://www.doi.org/" + doiurl)
                    time.sleep(10)

                    new_url = driver.current_url
                    if new_url is None:
                        new_url = ""
                    print("--------->", new_url)
                    df.at[index, "landing_url"] = new_url
                    df.to_parquet("db/postURL_papers.parquet", index=False)

                    if "springer.com" in new_url:
                        springer_bot(row["paperId"])

                    if "acm.org" in new_url:  # works well
                        acm_bot(row["paperId"])

                    if "ieee.org" in new_url:  # works well
                        ieee_bot(row["paperId"])

                    if "spiedigitallibrary.org" in new_url:  # works well
                        spie_bot(row["paperId"])

                    if "aip.org" in new_url:
                        aip_bot(row["paperId"])

                    if "iop.org" in new_url:
                        iop_bot(row["paperId"])

                    cleanup()

                except:
                    pass

if 0:
    # df.loc[df['landing_url'].str.contains('validate.perfdrive.com', na=False), 'landing_url'] = '' # or np.nan
    ids = df[df["landing_url"].str.contains("optica.org", case=False, na=False)].index
    print("len of selected rows: ", len(ids))

    for index in ids:
        row = df.loc[index]
        file_exist = check_file_exist(row["paperId"])

        if not file_exist:
            driver.get(row["landing_url"])
            time.sleep(10)

            if "nature.com" in row["landing_url"]:
                nature_bot(row["paperId"])

            if "springer.com" in row["landing_url"]:
                springer_bot(row["paperId"])

            if "acm.org" in row["landing_url"]:  # works well
                acm_bot(row["paperId"])

            if "ieee.org" in row["landing_url"]:  # works well
                ieee_bot(row["paperId"])

            if "spiedigitallibrary.org" in row["landing_url"]:  # works well
                spie_bot(row["paperId"])

            if "aip.org" in row["landing_url"]:  # works well
                aip_bot(row["paperId"])

            if "iop.org" in row["landing_url"]:
                iop_bot(row["paperId"])

            # if 'wiley.com' in row['landing_url']:
            #     wiley_bot(row['paperId'])

            if "optica.org" in row["landing_url"]:
                optica_bot(row["paperId"])

            # sys.exit()
            cleanup()


print("====== DONE! ======")
driver.quit()
