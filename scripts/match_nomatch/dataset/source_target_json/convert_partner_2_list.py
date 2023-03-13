# %%
import json
import pandas as pd


# %%
with open('partner_list_clean_final_eff_gcv_paddle_easy.json') as f:
    partner_list = json.load(f)

df = pd.DataFrame(partner_list)

# %%

df.to_csv('partner_list_clean_final_eff_gcv_paddle_easy.csv')

# %%
