import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from time import sleep
import pandas as pd

from crawl import crawl_phone
from crawl import crawl_laptop

if __name__ == "__main__":
    chrome_options= Options()
    service = Service(executable_path=r"D:\BAO KHANH\Chatbot-FPTshop\chromedriver-win64\chromedriver-win64\chromedriver.exe")
    chrome_options = webdriver.ChromeOptions()
    # chrome_options.add_argument('--headless') 
    chrome_options.add_argument('--no-sandbox') 
    chrome_options.add_argument('--disable-dev-shm-usage') 
    chrome_options.add_argument("start-maximized") 
    chrome_options.add_argument("disable-infobars") 
    chrome_options.add_argument("--disable-extensions")

    driver_1= webdriver.Chrome(service=service,options=chrome_options)
    driver_2= webdriver.Chrome(service=service,options=chrome_options)
    url_1= 'https://fptshop.com.vn/may-tinh-xach-tay'
    url_2= 'https://fptshop.com.vn/may-tinh-xach-tay'
    driver_1.get(url_1)
    driver_2.get(url_2)

    info_phone_name = crawl_phone()