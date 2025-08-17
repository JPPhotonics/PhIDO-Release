import asyncio
from urllib.parse import urlparse

from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright


def extract_main_domain(url):
    # Parse the URL
    parsed_url = urlparse(url)

    # Get the netloc (network location), which includes the domain
    domain = parsed_url.netloc

    # Split the domain into parts
    domain_parts = domain.split(".")

    # Check if it's a standard domain like example.com, co.uk, etc.
    if len(domain_parts) > 2:
        # For domains like www.example.com, www.example.co.uk, etc.
        main_domain = ".".join(domain_parts[-2:])
    else:
        # For simple domains like example.com
        main_domain = domain

    return main_domain


def take_screenshot(url, output_file):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)

        # Capture the final URL after redirects
        final_url = page.url

        # Take a screenshot
        page.screenshot(path=output_file)

        browser.close()
        return final_url


async def save_page_as_html(url, file_name):
    async with async_playwright() as p:
        # Launch a browser (use 'chromium', 'firefox', or 'webkit')
        browser = await p.chromium.launch()
        page = await browser.new_page()

        # Navigate to the URL
        await page.goto(url)

        # Get the page content (including HTML and potentially inlined resources)
        html_content = await page.content()

        # Save the HTML content to a file
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Close the browser
        await browser.close()


if 1:
    asyncio.run(
        save_page_as_html("https://www.doi.org/10.1117/12.2255794", "example.html")
    )


if 0:  # save screenshot
    final_url = take_screenshot(
        "https://www.doi.org/10.1117/12.2255794", "screenshot---.png"
    )
    print(f"landing page: {final_url}")
    print()
    main_domain = extract_main_domain(final_url)
    print("main domain:", main_domain)
