import os
import sys
import json
import numpy as np
import json
import skimage
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import colors
from openpyxl.styles import Font, Color, Alignment, PatternFill
from openpyxl.utils import get_column_letter

# These are the imports from other files
from transferBbox import *
from evaluate import *



# You need to have bboxResults initialised as a dictionary of numpy arrays.
# bboxCFList is a list of numpy arrays in the same order as bboxResults

# Format your bbox and confidence level results (if needed) and append them in these variables
bboxResults = {}
bboxCFList = []


# First convert dictionary of lists into a dataframe, with the image name as index
df = pd.DataFrame.from_dict(bboxResults,orient='index')
df.columns = ['BBox Array']
# Convert list of Confidence Level numpy array into pandas Series and append into the dataframe
cfdf = pd.Series(bboxCFList)
df = df.assign(CF=cfdf.values)
print(df)


try: 
    df.to_excel(r'exports/results.xlsx', header=True)
except PermissionError:
    print("There is already an existing file. You should remove it if you want to overwrite the file.")


# First transfer raw bounding box values and confidence levels of each image into an excel sheet
labelFilePath = r'exports/_label_nomask.json'
transfer_bbox(labelFilePath)

# Set the column names of each column used
set_header()

# Calculate and store each evaluation matrix
totalImg = calculate_total_images(bboxResults)
fNegCount = calculate_false_negatives(bboxResults)
fPosCount = calculate_false_positives(bboxResults)
passingRate = calculate_passing_rate()
average_cf = calculate_avg_cf()

# Finally, clean up the excel sheet for easy reading
clean_excel()