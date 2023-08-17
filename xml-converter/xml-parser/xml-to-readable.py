import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import re

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
file = askopenfilename() # show an "Open" dialog box and return the path to the selected file

parseComments = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
tree = ET.parse(file, parseComments)
root = tree.getroot()
xmlMap = {c: p for p in tree.iter() for c in p }
nextHasComment = False

class Depth:                     # The target object of the parser
    maxDepth = 0
    depth = 0
    def start(self, tag, attrib):   # Called for each opening tag.
        self.depth += 1
        if self.depth > self.maxDepth:
            self.maxDepth = self.depth
    def end(self, tag):             # Called for each closing tag.
        self.depth -= 1
    def data(self, data):
        pass            # We do not need to do anything with data.
    def close(self):    # Called when all data has been parsed.
        return self.maxDepth

parseDepth = ET.XMLParser(target=Depth())
parseDepth.feed(ET.tostring(root))
maxDepth = parseDepth.close()

def getText(node):
    try:
        match = re.search('(?:{.*?})(.*)', node)
        text = match.group(1)
    except:
        text = node
    return text

def getColumn(node):
    counter = 0
    newValue = node
    while newValue != None:
        try:
            newValue = xmlMap.get(newValue)
            counter += 1
        except:
            break
    return counter

def file_save(file, columns):
    exportName= os.path.splitext(os.path.basename(file))[0]
    #desktopPath = os.path.expanduser("~/Desktop")

    try:
        directory = askdirectory()
            #df.to_excel(f.name, index=False)
        fileName = directory+'/'+exportName+'.xlsx'
        with pd.ExcelWriter(fileName, engine="xlsxwriter") as writer:   
            df.to_excel(writer, sheet_name="Sheet1", index=False)
            workbook = writer.book
            worksheet = writer.sheets["Sheet1"]
            for idx, col in enumerate(df):  # loop through all columns
                series = df[col]
                max_len = max((
                    series.astype(str).map(len).max(),  # len of largest item
                    len(str(series.name))  # len of column name/header
                    )) + 1  # adding a little extra space
                print(max_len)
                if max_len > 20:
                    wrap = workbook.add_format({'text_wrap' : True})
                    worksheet.set_column(idx, idx, 20, wrap)
                else:
                    worksheet.set_column(idx, idx, max_len)        
            writer.close()
    except: 
        #f is None: # asksaveasfile return `None` if dialog closed with "cancel".
        return

for index, node in enumerate(root.iter()):
    nodeText = getText(node.tag)
    column = getColumn(node)
    neededColumns = maxDepth + 1
    
    if "function Comment" in str(node):
        nodeComment = node.text[:-1]
        nextHasComment = True
    elif index == 0:
        title = nodeText
        df = pd.DataFrame(columns=['Column {}'.format(x) for x in list(range(1,neededColumns))])
        defaultColumns = ['Annotation', 'Description']
        df[defaultColumns] = None
        df.style.set_caption(title)
        df.loc[len(df)] = {'Column {}'.format(column): nodeText, 'Annotation' : "", 'Description' : ""}
    else:
        df.loc[len(df)] = {'Column {}'.format(column): nodeText, 'Annotation' : "("+nodeComment+")", 'Description' : ""} \
        if (nextHasComment and index != 0) else {'Column {}'.format(column): nodeText, 'Annotation' : "", 'Description' : ""}
        nextHasComment = False

file_save(file, neededColumns)
