from pytablewriter import MarkdownTableWriter
from pytablewriter.style import Style
import sys
import pandas as pd

def args_to_md(model,args_dict):
    writer = MarkdownTableWriter()
    writer.table_name = model
    writer.headers=list(args_dict.keys())
    # print('headers: ',writer.headers)
    writer.value_matrix=[list(args_dict.values())]
    # print('value_matrix: ',writer.value_matrix)
    writer.column_styles = [Style(align="center") for _ in range(len(writer.headers))]
    print(writer.write_table())

def topics_to_md(model,topics_dict):
    writer = MarkdownTableWriter()
    writer.table_name = model
    writer.from_dataframe(
        pd.DataFrame(topics_dict),
        add_index_column=True,
    )
    writer.write_table()
