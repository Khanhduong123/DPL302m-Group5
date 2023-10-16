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