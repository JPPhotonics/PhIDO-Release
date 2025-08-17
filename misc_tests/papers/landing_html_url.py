import os

import pandas as pd
from papersdb import *

all_papers_df = pd.read_parquet("db/papers.parquet")
all_papers_df["landing_url"] = None

for index, row in all_papers_df.iterrows():
    externalIds = row["externalIds"]
    if externalIds["DOI"]:
        if os.path.exists(f"db/pdfs/{row['paperId']}.pdf"):
            # print('file exists!')
            pass
        else:
            url = "https://www.doi.org/" + externalIds["DOI"]
            print("____________")
            print("DOI:", url)
            html_content, landing_url = save_webpage(url, f"db/html/{row['paperId']}")
            print("landing:", landing_url)

            all_papers_df.at[index, "landing_url"] = landing_url

            all_papers_df.to_parquet("db/papers.parquet", index=False)

            # time.sleep(2)
