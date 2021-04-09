'''
openapi_cctv.py

경찰청 도시교통정보 센터 OPEN API : < http://www.utic.go.kr/>
'''

import os
import copy
import threading

import requests
import subprocess
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import pickle
import json

import argparse

import prograss_bar

# https://www.utic.go.kr:449/guide/imsOpenData.do?key=
# http://www.utic.go.kr/guide/cctvOpenData.do?key=SfjuzjsQMz2QHKtRqudqQpnc7zboULMUkD8UnjNsWItwZftRuoOLCTP9hxjc
# 106.246.235.109

EXIT_ROCK 	= threading.Lock()
WAIT_T		= 10

WAITING_TIME = 30
AUTH_KEY 	= 'YOUR KEY HERE'
API_URL 	= 'http://www.utic.go.kr/guide/cctvOpenData.do'

REQUEST_URL = (
	API_URL 
	+ "?key=" + AUTH_KEY
)

WEB_CHROME_DRIVER = '/usr/local/bin/chromedriver'

print("request url : " + REQUEST_URL);
print("connect...");

RESULT_URLS = []

def captureHLS():
	pass

def getUrls(save_file = None, list_file = None):
	if(list_file == None):
		return

	try:
		with open(list_file, "r") as json_file:
			list_file = json.load(json_file)

		driver_opt = webdriver.ChromeOptions()
		driver_opt.add_argument('headless')

		# driver = webdriver.Chrome(WEB_CHROME_DRIVER, options=driver_opt )
		driver = webdriver.Chrome(WEB_CHROME_DRIVER)

		driver.get(REQUEST_URL)
		# print(driver.page_source)

	
		cctv_anchor = driver.find_elements_by_css_selector("tr > td:nth-child(3) > a")
		train_cctv_anchor = []

		# < Search TRAIN_CCTV >
		for i, elem in enumerate(cctv_anchor) :
			cctv_name = elem.text.ljust(45, ' ')
			prograss_bar.print(i, 4767, 'Search:', cctv_name, 1, 25)
			for name in list_file :
				if elem.text.find(name) != -1 :
					train_cctv_anchor.append(elem)
					break;

		print("\nresult : ")
		print(train_cctv_anchor)

		# < Open result popup & get live stream url >
		print()
		for i, elem in enumerate(train_cctv_anchor) :
			driver.switch_to.window(driver.window_handles[0])
			
			cctv_name = elem.text
			cctv_name_just = cctv_name.ljust(45, ' ')
			prograss_bar.print(
				i, len(train_cctv_anchor), 
				'Scrapping URL:', cctv_name_just,
				1, 25
			)

			driver.execute_script(elem.get_attribute("href"))

			try:
				handles_before = driver.window_handles
				element = WebDriverWait(driver, WAITING_TIME).until(
					lambda driver: 
						(len(handles_before) != len(driver.window_handles)) 
				)
				driver.switch_to.window(driver.window_handles[-1])
				WebDriverWait(driver, WAITING_TIME).until(
					EC.presence_of_element_located((By.ID, "vid_html5_api"))
				)
				
			except:
				print("Can not open CCTV pop-up page")
				continue

			video = driver.find_element_by_css_selector("#vid_html5_api > source")
			RESULT_URLS.append(
				(cctv_name, video.get_attribute("src")))

			driver.close()

		# < Create Save File >
		print("saving ... ")

		path = "./save"
		if not os.path.isdir(path):                                                           
			os.mkdir(path)

		with open('./save/'+save_file, 'wb') as file:
			pickle.dump(RESULT_URLS, file)

		print("Save Selenium's Tag Objects at ./save/result_urls.pickle")

	except Exception as err : 
		print("error : ", err)
	finally:
		print("[ quit driver ]")
		driver.quit()

def extract(name, url, start_naming_num):
	pre_t = 0

	path = "./sample/"+name
	if not os.path.isdir(path):                                                           
		os.mkdir(path)

	while True:
		if not EXIT_ROCK.locked():
			break

		time.sleep(WAIT_T)
		
		start_t = subprocess.check_output('ffmpeg -i "' + url + '" 2>&1 | grep start:', shell=True).decode("utf-8")
		start_t = start_t[start_t.find('start:'):-1]
		start_t = float(start_t[start_t.find(':')+1:start_t.find(',')])

		if pre_t != start_t:
			pre_t = start_t

			os.system(
				'ffmpeg -i "' + url + '"' + 
				' -ss 00:00:01 -vframes 1 -vsync 0 -q:v 2 ' + 
				'./sample/' + name + '/' + name + '-' + str(start_naming_num) + '.jpg' +
				' > /dev/null'
			)
			print(
				"name : ", name, 
				" / saved at : ", './sample/' + name + '/' + name + '-' + str(start_naming_num) + '.jpg'
			)

def pressKeyToExit():
	EXIT_ROCK.acquire()
	os.system('read -s -n 1 -p "Press any key to continue..."')
	print()
	EXIT_ROCK.release()

def loadAndExtract(save_file):
	try:
		data = []
		thread_list = []

		thread_list.append(
			threading.Thread(target=pressKeyToExit, args=())
		)

		with open('./save/'+save_file, 'rb') as f:
			data = pickle.load(f)

			for d in data:
				thread_list.append(
					threading.Thread(target=extract, args=(d[0], d[1], 0))
				)

		for t in thread_list:
			t.start()

	except Exception as err:
		print("error : ", err)

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--scan", dest="scan", action="store", help="List of CCTV name for Scanning")
parser.add_argument("-l", "--load", dest="load", action="store", help="Load scan save file")
args = parser.parse_args()

if args.scan:
	getUrls(save_file=args.load, list_file=args.scan)
	loadAndExtract(save_file="result_urls.pickle")

if args.load:
	loadAndExtract(save_file=args.load)