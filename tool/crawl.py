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

def get_laptop_info():
    URL_all_laptop_page = GetURL()[:6]
    data= {
        'laptop_names': [],
        'laptop_ids': [],
        'laptop_moneys': [],
        'laptop_chips': [],
        'laptop_memories': [],
        'laptop_storages': [],
        'laptop_video_cards':[]
    }
    for i in URL_all_laptop_page:
        try:
            driver.get(i)
            sleep(1)
            page_source= BeautifulSoup(driver.page_source,'html.parser')
            info_laptop_name = page_source.find('div',{'class':'l-pd-top'})
            laptop_name = info_laptop_name.find('h1',{'class':'st-name'}).get_text().strip()
            laptop_id = info_laptop_name.find('span',{'class':'st-sku'}).get_text().strip()

            info_laptop_money= page_source.find('div',{'class':'st-price__left'})
            laptop_money= info_laptop_money.find('div',{'class':'st-price-main'}).get_text().replace('₫','').strip()

            info_laptop= page_source.find('div',{'class':'st-param'})

            laptop_chip= info_laptop.find('li',{'data-info':'CPU'}).get_text().replace(',','').strip()
            laptop_memory = info_laptop.find('li',{'data-info':'RAM'}).get_text().replace('GB','').replace(',','').strip()
            laptop_screen = info_laptop.find('li',{'data-info':'Màn hình'}).get_text().replace(',','').strip()
            laptop_storage= info_laptop.find('li',{'data-info':'Ổ cứng'}).get_text().replace('GB','').strip()
            laptop_video_card= info_laptop.find('li',{'data-info':'Đồ họa'}).get_text().strip()
            
            
            data['laptop_names'].append(laptop_name)
            data['laptop_ids'].append(laptop_id)
            data['laptop_moneys'].append(laptop_money)
            data['laptop_chips'].append(laptop_chip)
            data['laptop_memories'].append(laptop_memory)
            data['laptop_storages'].append(laptop_storage)
            data['laptop_video_cards'].append(laptop_video_card)
            
            count +=1
            if count == 86 :
                break
        except:
            pass
    return data


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