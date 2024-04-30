import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from time import sleep
import pandas as pd


def GetURL():
    page_source = BeautifulSoup(driver.page_source)
    profiles= page_source.find_all('a',class_="cdt-product__name")
    all_URLs_profile = []
    for profile in profiles:
        profile_ID= profile.get('href')
        profile_URL= 'https://fptshop.com.vn/' + profile_ID
        if profile_URL not in all_URLs_profile:
            all_URLs_profile.append(profile_URL)
    return all_URLs_profile