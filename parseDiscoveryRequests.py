#import PyPDF2
import re
from sklearn.metrics.pairwise import cosine_similarity
import fitz
import numpy as np
import copy
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
model = SentenceTransformer("nlpaueb/legal-bert-small-uncased")
#model = SentenceTransformer("all-mpnet-base-v2")

input_dir = "RFA"

input_pdf_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".pdf")
    ]
)

def pdf2Text_fitz(filename):
    doc = fitz.open(filename)    # creating a pdf reader object
    text = ""
    for page in doc:
        text+=page.get_text().lower()
        # extracting text from page
    # closing the pdf file object
    doc.close()
    #remove everything after certificate of service, marks the end of requests
    m = re.sub(r"certificate of service.*$","",repr(text))
    #m = re.findall(r"[0-9]+\.\s.+?(?=\.\"*\'*\s*\\n)", m)
    m = re.findall(r"[0-9]+\.[\\n\s]+.+?(?=\.\"*\'*\s*\\n)", m)
    return [re.sub(r'^[0-9]+\.\s*', '', a).replace("\\n","") for a in m]

def pdf2Text(filename):
    pdfFileObj = open(filename, 'rb')
    # creating a pdf reader object
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    text = ""
    for i in range(pdfReader.numPages):
        pageObj = pdfReader.getPage(i)
        text += pageObj.extractText()
        # extracting text from page
    # closing the pdf file object
    m = re.findall(r"[0-9]+\.\s.+?(?=\.\s*\\n)", repr(text))
    pdfFileObj.close()
    return m

def getPairs(txt1,txt2,cossim):
    maxcos = cossim.argmax(axis=1)
    maxcosInv = cossim.argmax(axis=0)
    nomatch = maxcosInv[maxcos] != np.arange(maxcos.shape[0])
    nomatchInv = maxcos[maxcosInv] != np.arange(maxcosInv.shape[0])
    pairs=[]
    for i,j in enumerate(maxcos):
        pairs.append([txt1[i],txt2[j]])
    return pairs
        
#TODO save vstack and np.append to one call at the end of the method
def consolidate(txt1,txt2,txt1_embed,txt2_embed,cossim,counts):
    masterlist = copy.deepcopy(txt1)
    maxcos = cossim.argmax(axis=1)
    maxcosInv = cossim.argmax(axis=0)
    nomatch = maxcosInv[maxcos] != np.arange(maxcos.shape[0])
    nomatchInv = maxcos[maxcosInv] != np.arange(maxcosInv.shape[0])
    seen = []
    for i in range(counts.shape[0]):
        if not nomatch[i]:
            counts[i] += 1
            seen += [maxcos[i]]
        else:
            if maxcos[i] not in seen:
                masterlist.append(txt2[maxcos[i]])
                counts = np.append(counts,[1])
                txt1_embed = np.vstack((txt1_embed,txt2_embed[maxcos[i]]))
                seen += [maxcos[i]]
    set2 = set(range(len(txt2)))
    set2remaining = set2 - set(seen)
    
    print(set2remaining)
    for i in set2remaining:
        masterlist.append(txt2[i])
        txt1_embed = np.vstack((txt1_embed,txt2_embed[i]))
        counts = np.append(counts,[1])
    return masterlist,txt1_embed,counts

def printbycount(requests,counts):
    order = counts.argsort()
    order = order[::-1]
    for i in order:
        print("count:",counts[i],"\n",requests[i],"\n")
        
def writebycount(requests,counts):
    order = counts.argsort()
    order = order[::-1]
    with open('request_ranks.txt', 'w') as f:
        for i in order:
            f.write("count: "+str(counts[i])+" \n "+requests[i]+" \n")        
    
def countRequests(filenames, csv=None):
    #start from scratch
    if csv is None:
        requests = pdf2Text_fitz(filenames[0])
        requests_embed = model.encode(requests)
        request_counts = np.ones(shape=len(requests))
        for i in range(1,len(filenames)):
            print(i,filenames[i])
            try:
                newRequests = pdf2Text_fitz(filenames[i])
                newRequests_embed = model.encode(newRequests)
                if len(newRequests_embed) <= 1:
                    print("embeddings error, skipping")
                    raise Exception()
            except Exception as e:
                print(e)
                continue
            print("shapes", requests_embed.shape, newRequests_embed.shape,len(newRequests_embed))
            similarities = cosine_similarity(requests_embed,newRequests_embed)
            requests,requests_embed,request_counts = consolidate(requests,newRequests,requests_embed,newRequests_embed,similarities,request_counts)

        return requests,request_counts

def getEmbeds(filenames,csv=None):
    requests = pdf2Text_fitz(filenames[0])
    embeds = model.encode(requests)
    for i in range(1,len(filenames)):
        #print(len(requests),embeds.shape)        
        #print(i,filenames[i])
        
        try:
            newRequests = pdf2Text_fitz(filenames[i])
            newRequests_embed = model.encode(newRequests)
            if len(newRequests_embed) <= 1:
                print("embeddings error, skipping",filenames[i])
                raise Exception()
        except Exception as e:
            print(e)
            continue    
        requests+=newRequests
        embeds = np.vstack((embeds,newRequests_embed))
    return requests, np.array(embeds)


if __name__ == '__main__':
    # creating a pdf file object
    #pdfFileObj = open('.\\discoveries\\ANDREW J. GORMAN & ASSOCIATES.pdf', 'rb')
    pdf1 = '.\\discoveries\\BUCHANAN & BUCHANAN, P.A..pdf'
    pdf2 = '.\\discoveries\\Evangelo, Brandt & Lippert, P.A.-2.pdf'
    pdf3 = '.\\discoveries\\ANDREW J. GORMAN & ASSOCIATES.pdf'

    pdfs = [pdf1,pdf2,pdf3]
    pdfs = input_pdf_paths
    #r,e = getEmbeds(pdfs)

    requests,counts = countRequests(pdfs)

    pdData = pd.DataFrame({"request":requests, "count":counts})
    sortedpd = pdData.sort_values("count",ascending=False)
    sortedpd.to_csv("rfa_request_ranks.csv",index=False)

    # nonum_text1 = pdf2Text_fitz(pdf1)
    # nonum_text2 = pdf2Text_fitz(pdf2)
    # nonum_text3 = pdf2Text_fitz(pdf3)

    # ## nonum_text1 = [re.sub(r'^[0-9]+\.\s+', '', a).replace("\\n","") for a in text1]
    # ## nonum_text2 = [re.sub(r'^[0-9]+\.\s+', '', a).replace("\\n","") for a in text2]
    # ## nonum_text3 = [re.sub(r'^[0-9]+\.\s+', '', a).replace("\\n","") for a in text3]
    # #
    # ## text1 = [a.replace("\\n","") for a in text1]
    # ## text2 = [a.replace("\\n","") for a in text2]
    # ## text3 = [a.replace("\\n","") for a in text3]

    # text1_embed = model.encode(nonum_text1)
    # text2_embed = model.encode(nonum_text2)
    # text3_embed = model.encode(nonum_text3)


    # sim12 = cosine_similarity(text1_embed,text2_embed)

    # ## pairs12 = getPairs(text1,text2,sim12)
    # ## #pairs13 = getPairs(text1,text3,sim13)
    # #
    # ## with open('pairs.txt', 'w') as f:
    # ##     for i in range(len(pairs12)):
    # ##  #       f.write(pairs12[i][0]+" \n "+pairs12[i][1]+" \n "+pairs13[i][1]+"\n\n\n")
    # ##         f.write(pairs12[i][0]+" \n "+pairs12[i][1]+" \n "+"\n\n\n")

    # counts = np.ones(shape=len(nonum_text1))


    # requests,counts = consolidate(nonum_text1,nonum_text2,sim12,counts)
    # requests_embed = model.encode(requests)
    # sim = cosine_similarity(requests_embed,text3_embed)
    # requests,counts = consolidate(requests,nonum_text3,sim,counts)

    # ## printbycount(requests,counts)
    # ## writebycount(requests,counts)

    # pdData = pd.DataFrame({"request":requests, "count":counts})
    # sortedpd = pdData.sort_values("count",ascending=False)
    # sortedpd.to_csv("request_ranks.csv",index=False)