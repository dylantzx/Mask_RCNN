def transfer_bbox(labelFilePath):
    
    # Load excel sheet for manipulation
    wb = load_workbook(filename=r'exports/results.xlsx')
    ws = wb.active
    
    rowNum = 2      # First entry is at row number 2
    
    # Load json file 
    with open(labelFilePath) as lf:
        lfData = json.load(lf)
        
    # Transfer bounding box values row by row
    for image in lfData["annotations"]:
        
        # Ground Truth bounding box list is in different format than mask RCNN rois
        gtList = [round(num) for num in image["bbox"]]
        
        # Reformats the list following mask RCNN rois for easier comparison
        if (len(gtList) == 4):
            reformattedList = [gtList[1], gtList[0], gtList[1]+gtList[3], gtList[0]+gtList[2]]
            
            # Check for negative numbers and bump them up to 0
            for i in range(len(reformattedList)):
                if reformattedList[i] < 0:
                    reformattedList[i] = 0
            
            # Afterwards convert to numpy array
            npArray = np.asarray(reformattedList)
            
        ws.cell(row=rowNum, column=16, value=str(npArray))
        rowNum +=1
        
    # Update excel sheet
    wb.save(filename=r'exports/results.xlsx')
    
    return