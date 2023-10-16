import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from time import sleep
import pandas as pd
from geturl import *
print('- Finish importing packages')




def get_phone_info():
    URLs_all_page = GetURL()
    data= dict(phone_names = [],phone_ids=[],phone_money = [],phone_chips=[],phone_memories=[],phone_screens=[],camera_selfies=[])
    count = 0
    for i in URLs_all_page:
        try:
            driver.get(i)
            sleep(2)
            page_source= BeautifulSoup(driver.page_source,"html.parser")
            #name of phone
            info_phone_name= page_source.find('div',{'class':'l-pd-top'})
            phone_name = info_phone_name.find('h1',{'class':'st-name'}).get_text().strip()
            phone_id = info_phone_name.find('span',{'class':'st-sku'}).get_text().strip()
            #money of this phone
            info_phone_monney = page_source.find('div',{'class':'st-price__left'})
            phone_money = info_phone_monney.find('div',{'class':'st-price-main'}).get_text().replace('₫','').strip()
            #info about this phone
            info_phone= page_source.find('div',{'class':'st-param'})
            
            phone_chip = info_phone.find('li',{'data-info':'CPU'}).get_text().strip()# CPU
            phone_memory = info_phone.find('li',{'data-info':'Bộ nhớ trong'}).get_text().replace('GB','').strip()#memory
            phone_screen = info_phone.find('li',{'data-info':'Màn hình'}).get_text().replace('Chính:','').lstrip().split(',')[0]#screen
            camera_selfie= info_phone.find('li',{'data-info':'Camera Selfie'}).get_text().strip()#camera selfie
            
            #append data to the list
            data['phone_names'].append(phone_name)
            data['phone_ids'].append(phone_id)
            data['phone_money'].append(int(phone_money.replace('.','')))
            data['phone_chips'].append(phone_chip)
            data['phone_memories'].append(int(phone_memory))
            data['phone_screens'].append(phone_screen)
            data['camera_selfies'].append(camera_selfie)
            count +=1
            if count == 86 :
                break
        except:
            pass
    return data